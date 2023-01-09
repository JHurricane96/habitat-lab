#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode as CN
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_generator import ISLAND_RADIUS_LIMIT
from habitat.datasets.rearrange.geometry_utils import direction_to_quaternion, get_bb

import habitat.datasets.rearrange.samplers as samplers
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import habitat.sims.habitat_simulator.sim_utilities as sutils
import habitat_sim
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.rearrange.samplers.receptacle import (
    OnTopOfReceptacle,
    ReceptacleSet,
    ReceptacleTracker,
    find_receptacles,
)
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat.tasks.nav.object_nav_task import ObjectViewLocation
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.common import cull_string_list_by_substrings
from habitat_sim.nav import NavMeshSettings
from habitat_sim.agent.agent import ActionSpec
from habitat_sim.agent.controls import ActuationSpec
from habitat_sim.utils.common import quat_to_coeffs


def get_sample_region_ratios(load_dict) -> Dict[str, float]:
    sample_region_ratios: Dict[str, float] = defaultdict(lambda: 1.0)
    sample_region_ratios.update(
        load_dict["params"].get("sample_region_ratio", {})
    )
    return sample_region_ratios


class RearrangeEpisodeGenerator:
    """Generator class encapsulating logic for procedurally sampling individual episodes for general rearrangement tasks.

    Initialized from a provided configuration file defining dataset paths, object,scene,and receptacle sets, and state sampler parameters.

    See rearrange_dataset.py for details on the RearrangeDataset and RearrangeEpisodes produced by this generator.
    See this file's main executable function below for details on running the generator.
    See `test_rearrange_episode_generator()` in test/test_rearrange_task.py for unit test example.
    """

    def __enter__(self) -> "RearrangeEpisodeGenerator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.sim != None:
            self.sim.close(destroy=True)
            del self.sim

    def __init__(
        self,
        cfg: CN,
        debug_visualization: bool = False,
        limit_scene_set: Optional[str] = None,
    ) -> None:
        """
        Initialize the generator object for a particular configuration.
        Loads yaml, sets up samplers and debug visualization settings.
        """
        # load and cache the config
        self.cfg = cfg
        self.start_cfg = self.cfg.clone()
        self._limit_scene_set = limit_scene_set

        # debug visualization settings
        self._render_debug_obs = self._make_debug_video = debug_visualization
        self.vdb: DebugVisualizer = (
            None  # visual debugger initialized with sim
        )

        # hold a habitat Simulator object for efficient re-use
        self.sim: habitat_sim.Simulator = None
        # initialize an empty scene and load the SceneDataset
        self.initialize_sim("NONE", self.cfg.dataset_path)

        # Setup the sampler caches from config
        self._get_resource_sets()
        self._get_scene_sampler()
        self._get_obj_samplers()
        self._get_ao_state_samplers()

        # cache objects sampled by this generator for the most recent episode
        self.ep_sampled_objects: List[
            habitat_sim.physics.ManagedRigidObject
        ] = []
        self.num_ep_generated = 0

    def _get_resource_sets(self) -> None:
        """
        Extracts and validates scene, object, and receptacle sets from the config and fills internal datastructures for later reference.
        Assumes the Simulator (self.sim) is already initialized.
        """
        # {scene set name -> [scene handles]}
        self._scene_sets: Dict[str, List[str]] = {}

        # {object set name -> [object handles]}
        self._obj_sets: Dict[str, List[str]] = {}

        # {receptacle set name -> ([included object handles], [excluded object handles], [included receptacle name substrings], [excluded receptacle name substrings])}
        self._receptacle_sets: Dict[str, ReceptacleSet] = {}

        expected_list_keys = ["included_substrings", "excluded_substrings"]
        # scene sets
        for scene_set in self.cfg.scene_sets:
            assert "name" in scene_set
            assert (
                scene_set["name"] not in self._scene_sets
            ), f"cfg.scene_sets - Duplicate name ('{scene_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in scene_set
                ), f"Expected list key '{list_key}'."
                assert (
                    type(scene_set[list_key]) is list
                ), f"cfg.scene_sets - '{scene_set['name']}' '{list_key}' must be a list of strings."
            self._scene_sets[
                scene_set["name"]
            ] = cull_string_list_by_substrings(
                self.sim.metadata_mediator.get_scene_handles(),
                scene_set["included_substrings"],
                scene_set["excluded_substrings"],
            )

        # object sets
        for object_set in self.cfg.object_sets:
            assert "name" in object_set
            assert (
                object_set["name"] not in self._obj_sets
            ), f"cfg.object_sets - Duplicate name ('{object_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in object_set
                ), f"Expected list key '{list_key}'."
                assert (
                    type(object_set[list_key]) is list
                ), f"cfg.object_sets - '{object_set['name']}' '{list_key}' must be a list of strings."
            self._obj_sets[
                object_set["name"]
            ] = cull_string_list_by_substrings(
                self.sim.get_object_template_manager().get_template_handles(),
                object_set["included_substrings"],
                object_set["excluded_substrings"],
            )

        # receptacle sets
        expected_list_keys = [
            "included_object_substrings",
            "excluded_object_substrings",
            "included_receptacle_substrings",
            "excluded_receptacle_substrings",
        ]
        for receptacle_set in self.cfg.receptacle_sets:
            assert "name" in receptacle_set
            assert (
                receptacle_set["name"] not in self._receptacle_sets
            ), f"cfg.receptacle_sets - Duplicate name ('{receptacle_set['name']}') detected."
            for list_key in expected_list_keys:
                assert (
                    list_key in receptacle_set
                ), f"Expected list key '{list_key}'."
                assert (
                    type(receptacle_set[list_key]) is list
                ), f"cfg.receptacle_sets - '{receptacle_set['name']}' '{list_key}' must be a list of strings."

            self._receptacle_sets[receptacle_set["name"]] = ReceptacleSet(
                **receptacle_set
            )

    def _get_obj_samplers(self) -> None:
        """
        Extracts object sampler parameters from the yaml config file and constructs the sampler objects.
        """
        self._obj_samplers: Dict[str, samplers.ObjectSampler] = {}

        for obj_sampler_info in self.cfg.object_samplers:
            assert "name" in obj_sampler_info
            assert "type" in obj_sampler_info
            assert "params" in obj_sampler_info
            assert (
                obj_sampler_info["name"] not in self._obj_samplers
            ), f"Duplicate object sampler name '{obj_sampler_info['name']}' in config."
            if obj_sampler_info["type"] == "uniform":
                assert "object_sets" in obj_sampler_info["params"]
                assert "receptacle_sets" in obj_sampler_info["params"]
                assert "num_samples" in obj_sampler_info["params"]
                assert "orientation_sampling" in obj_sampler_info["params"]
                # merge and flatten object and receptacle sets
                object_handles = [
                    x
                    for y in obj_sampler_info["params"]["object_sets"]
                    for x in self._obj_sets[y]
                ]
                object_handles = sorted(set(object_handles))
                if len(object_handles) == 0:
                    raise ValueError(
                        f"Found no object handles for {obj_sampler_info}"
                    )

                self._obj_samplers[
                    obj_sampler_info["name"]
                ] = samplers.ObjectSampler(
                    object_handles,
                    obj_sampler_info["params"]["receptacle_sets"],
                    (
                        obj_sampler_info["params"]["num_samples"][0],
                        obj_sampler_info["params"]["num_samples"][1],
                    ),
                    obj_sampler_info["params"]["orientation_sampling"],
                    get_sample_region_ratios(obj_sampler_info),
                    obj_sampler_info["params"].get(
                        "nav_to_min_distance", -1.0
                    ),
                    obj_sampler_info["params"].get("sample_probs", None),
                )
            else:
                logger.info(
                    f"Requested object sampler '{obj_sampler_info['type']}' is not implemented."
                )
                raise (NotImplementedError)

    def _get_object_target_samplers(self) -> None:
        """
        Initialize target samplers. Expects self.episode_data to be populated by object samples.
        """

        self._target_samplers: Dict[str, samplers.ObjectTargetSampler] = {}
        for target_sampler_info in self.cfg.object_target_samplers:
            assert "name" in target_sampler_info
            assert "type" in target_sampler_info
            assert "params" in target_sampler_info
            assert (
                target_sampler_info["name"] not in self._target_samplers
            ), f"Duplicate target sampler name '{target_sampler_info['name']}' in config."
            if target_sampler_info["type"] == "uniform":
                # merge and flatten receptacle sets

                self._target_samplers[
                    target_sampler_info["name"]
                ] = samplers.ObjectTargetSampler(
                    # Add object set later
                    [],
                    target_sampler_info["params"]["receptacle_sets"],
                    (
                        target_sampler_info["params"]["num_samples"][0],
                        target_sampler_info["params"]["num_samples"][1],
                    ),
                    target_sampler_info["params"]["orientation_sampling"],
                    get_sample_region_ratios(target_sampler_info),
                    target_sampler_info["params"].get(
                        "nav_to_min_distance", -1.0
                    ),
                )
            else:
                logger.info(
                    f"Requested target sampler '{target_sampler_info['type']}' is not implemented."
                )
                raise (NotImplementedError)

    def _get_scene_sampler(self) -> None:
        """
        Initialize the scene sampler.
        """
        self._scene_sampler: Optional[samplers.SceneSampler] = None
        if self.cfg.scene_sampler.type == "single":
            self._scene_sampler = samplers.SingleSceneSampler(
                self.cfg.scene_sampler.params.scene
            )
        elif self.cfg.scene_sampler.type == "subset":
            unified_scene_set: List[str] = []
            # concatenate all requested scene sets
            for set_name in self.cfg.scene_sampler.params.scene_sets:
                if (
                    self._limit_scene_set is not None
                    and set_name != self._limit_scene_set
                ):
                    continue
                assert (
                    set_name in self._scene_sets
                ), f"'subset' SceneSampler requested scene_set name, '{set_name}', not found."
                unified_scene_set += self._scene_sets[set_name]

            # cull duplicates
            unified_scene_set = sorted(set(unified_scene_set))
            self._scene_sampler = samplers.MultiSceneSampler(unified_scene_set)
        else:
            logger.error(
                f"Requested scene sampler '{self.cfg.scene_sampler.type}' is not implemented."
            )
            raise (NotImplementedError)

    def _get_ao_state_samplers(self) -> None:
        """
        Initialize and cache all ArticulatedObject state samplers from configuration.
        """
        self._ao_state_samplers: Dict[
            str, samplers.ArticulatedObjectStateSampler
        ] = {}
        for ao_info in self.cfg.ao_state_samplers:
            assert "name" in ao_info
            assert "type" in ao_info
            assert "params" in ao_info
            assert (
                ao_info["name"] not in self._ao_state_samplers
            ), f"Duplicate AO state sampler name {ao_info['name']} in config."

            if ao_info["type"] == "uniform":
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.ArticulatedObjectStateSampler(
                    ao_info["params"][0],
                    ao_info["params"][1],
                    (ao_info["params"][2], ao_info["params"][3]),
                )
            elif ao_info["type"] == "categorical":
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.ArtObjCatStateSampler(
                    ao_info["params"][0],
                    ao_info["params"][1],
                    (ao_info["params"][2], ao_info["params"][3]),
                )
            elif ao_info["type"] == "composite":
                composite_ao_sampler_params: Dict[
                    str, Dict[str, Tuple[float, float, bool]]
                ] = {}
                for entry in ao_info["params"]:
                    ao_handle = entry["ao_handle"]
                    should_sample_all_joints = entry.get(
                        "should_sample_all_joints", False
                    )
                    link_sample_params = entry["joint_states"]
                    assert (
                        ao_handle not in composite_ao_sampler_params
                    ), f"Duplicate handle '{ao_handle}' in composite AO sampler config."
                    composite_ao_sampler_params[ao_handle] = {}
                    for link_params in link_sample_params:
                        link_name = link_params[0]
                        assert (
                            link_name
                            not in composite_ao_sampler_params[ao_handle]
                        ), f"Duplicate link name '{link_name}' for handle '{ao_handle} in composite AO sampler config."
                        composite_ao_sampler_params[ao_handle][link_name] = (
                            link_params[1],
                            link_params[2],
                            should_sample_all_joints,
                        )
                self._ao_state_samplers[
                    ao_info["name"]
                ] = samplers.CompositeArticulatedObjectStateSampler(
                    composite_ao_sampler_params,
                    ao_info.get("apply_prob", None),
                )
            else:
                logger.error(
                    f"Requested AO state sampler type '{ao_info['type']}' not implemented."
                )
                raise (NotImplementedError)

    def _reset_samplers(self) -> None:
        """
        Reset any sampler internal state related to a specific scene or episode.
        """
        # self.ep_sampled_objects = []
        self._scene_sampler.reset()
        for sampler in self._obj_samplers.values():
            sampler.reset()

    def generate_scene(self) -> str:
        """
        Sample a new scene and re-initialize the Simulator.
        Return the generated scene's handle.
        """
        cur_scene_name = self._scene_sampler.sample()
        logger.info(f"Initializing scene {cur_scene_name}")
        self.initialize_sim(cur_scene_name, self.cfg.dataset_path)

        return cur_scene_name

    def visualize_scene_receptacles(self) -> None:
        """
        Generate a wireframe bounding box for each receptacle in the scene, aim the camera at it and record 1 observation.
        """
        logger.info("visualize_scene_receptacles processing")
        receptacles = find_receptacles(self.sim)
        for receptacle in receptacles:
            logger.info("receptacle processing")
            viz_objects = receptacle.add_receptacle_visualization(self.sim)

            # sample points in the receptacles to display
            # for sample in range(25):
            #     sample_point = receptacle.sample_uniform_global(self.sim, 1.0)
            #     sutils.add_viz_sphere(self.sim, 0.025, sample_point)

            if viz_objects:
                # point the camera at the 1st viz_object for the Receptacle
                self.vdb.look_at(
                    viz_objects[0].root_scene_node.absolute_translation
                )
                self.vdb.get_observation()
            else:
                logger.warning(
                    f"visualize_scene_receptacles: no visualization object generated for Receptacle '{receptacle.name}'."
                )

    def generate_episodes(
        self, num_episodes: int = 1, verbose: bool = False
    ) -> List[RearrangeEpisode]:
        """
        Generate a fixed number of episodes.
        """
        generated_episodes: List[RearrangeEpisode] = []
        failed_episodes = 0
        if verbose:
            pbar = tqdm(total=num_episodes)
        self.times = defaultdict(float)
        import time
        while len(generated_episodes) < num_episodes:
            t = time.process_time()
            new_episode = self.generate_single_episode()
            print("Tot time", time.process_time() - t)
            if new_episode is None:
                failed_episodes += 1
                continue
            generated_episodes.append(new_episode)
            if verbose:
                print("Generated NEW EPISODE! Num episodes:", len(generated_episodes))
                pbar.update(1)
        if verbose:
            pbar.close()

        logger.info(
            f"Generated {num_episodes} episodes in {num_episodes+failed_episodes} tries."
        )

        print(self.times)

        return generated_episodes
    
    def gen_cleanup(self, times):
        for k, v in times.items():
            self.times[k] += v
        print(times)

    def generate_single_episode(self) -> Optional[RearrangeEpisode]:
        """
        Generate a single episode, sampling the scene.
        """
        import time
        pt = time.process_time()
        times = {}

        # Reset the number of allowed objects per receptacle.
        recep_tracker = ReceptacleTracker(
            {k: v for k, v in self.cfg.max_objects_per_receptacle},
            self._receptacle_sets,
        )
        t = time.process_time()
        times["t1"] = t - pt
        pt = t

        self._reset_samplers()
        self.episode_data: Dict[str, Dict[str, Any]] = {
            "sampled_objects": {},  # object sampler name -> sampled object instances
            "sampled_targets": {},  # target sampler name -> (object, target state)
        }
        t = time.process_time()
        times["t2"] = t - pt
        pt = t

        ep_scene_handle = self.generate_scene()
        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))
        t = time.process_time()
        times["t3"] = t - pt
        pt = t


        scene_name = osp.basename(ep_scene_handle).split(".")[0]
        navmesh_path = osp.join(
            scene_base_dir, "navmeshes", scene_name + ".navmesh"
        )
        if osp.exists(navmesh_path):
            self.sim.pathfinder.load_nav_mesh(navmesh_path)
        else:
            self.sim.navmesh_settings = NavMeshSettings()
            self.sim.navmesh_settings.set_defaults()
            self.sim.navmesh_settings.agent_radius = 0.2
            self.sim.navmesh_settings.agent_height = 1.5
            # self.sim.navmesh_settings.agent_max_climb = agent_config.MAX_CLIMB
            self.sim.recompute_navmesh(self.sim.pathfinder, self.sim.navmesh_settings, include_static_objects=True)
            os.makedirs(osp.dirname(navmesh_path), exist_ok=True)
            self.sim.pathfinder.save_nav_mesh(navmesh_path)
        t = time.process_time()
        times["t4"] = t - pt
        pt = t


        self._get_object_target_samplers()
        target_numbers = {
            k: sampler.target_objects_number
            for k, sampler in self._target_samplers.items()
        }
        targ_sampler_name_to_obj_sampler_names = {}
        for targ_sampler_cfg in self.cfg.object_target_samplers:
            sampler_name = targ_sampler_cfg["name"]
            targ_sampler_name_to_obj_sampler_names[
                sampler_name
            ] = targ_sampler_cfg["params"]["object_samplers"]
        t = time.process_time()
        times["t5"] = t - pt
        pt = t


        target_receptacles = defaultdict(list)
        all_target_receptacles = []
        for sampler_name, num_targets in target_numbers.items():
            obj_sampler_name = targ_sampler_name_to_obj_sampler_names[
                sampler_name
            ][0]
            sampler = self._obj_samplers[obj_sampler_name]
            new_target_receptacles = []
            for _ in range(num_targets):
                new_receptacle = sampler.sample_receptacle(
                    self.sim, recep_tracker
                )
                if recep_tracker.update_receptacle_tracking(new_receptacle):
                    sampler.receptacle_candidates = None
                new_target_receptacles.append(new_receptacle)

            target_receptacles[obj_sampler_name].extend(new_target_receptacles)
            all_target_receptacles.extend(new_target_receptacles)
        t = time.process_time()
        times["t6"] = t - pt
        pt = t


        goal_receptacles = {}
        all_goal_receptacles = []
        for sampler, (sampler_name, num_targets) in zip(
            self._target_samplers.values(), target_numbers.items()
        ):
            new_goal_receptacles = []
            for _ in range(num_targets):
                new_receptacle = sampler.sample_receptacle(
                    self.sim,
                    recep_tracker,
                )
                if isinstance(new_receptacle, OnTopOfReceptacle):
                    new_receptacle.set_episode_data(self.episode_data)
                if recep_tracker.update_receptacle_tracking(new_receptacle):
                    sampler.receptacle_candidates = None

                new_goal_receptacles.append(new_receptacle)

            goal_receptacles[sampler_name] = new_goal_receptacles
            all_goal_receptacles.extend(new_goal_receptacles)
        t = time.process_time()
        times["t7"] = t - pt
        pt = t


        for recep in [*all_goal_receptacles, *all_target_receptacles]:
            recep_tracker.inc_count(recep.name)
        t = time.process_time()
        times["t8"] = t - pt
        pt = t


        # sample AO states for objects in the scene
        # ao_instance_handle -> [ (link_ix, state), ... ]
        ao_states: Dict[str, Dict[int, float]] = {}
        for _sampler_name, ao_state_sampler in self._ao_state_samplers.items():
            sampler_states = ao_state_sampler.sample(
                self.sim,
                [*all_target_receptacles, *all_goal_receptacles],
            )
            if sampler_states is None:
                times["t9"] = time.process_time() - t
                self.gen_cleanup(times)
                return None
            for sampled_instance, link_states in sampler_states.items():
                if sampled_instance.handle not in ao_states:
                    ao_states[sampled_instance.handle] = {}
                for link_ix, joint_state in link_states.items():
                    ao_states[sampled_instance.handle][link_ix] = joint_state
        self.ao_states = ao_states
        t = time.process_time()
        times["t9"] = t - pt
        pt = t


        # visualize after setting AO states to correctly see scene state
        if self._render_debug_obs:
            self.visualize_scene_receptacles()
            self.vdb.make_debug_video(prefix="receptacles_")
        t = time.process_time()
        times["t10"] = t - pt
        pt = t


        # sample object placements
        object_to_containing_receptacle = {}
        for sampler_name, obj_sampler in self._obj_samplers.items():
            object_sample_data = obj_sampler.sample(
                self.sim,
                recep_tracker,
                target_receptacles[sampler_name],
                snap_down=True,
                vdb=(self.vdb if self._render_debug_obs else None),
            )
            if len(object_sample_data) == 0:
                times["t11"] = time.process_time() - t
                self.gen_cleanup(times)
                return None
            new_objects, receptacles = zip(*object_sample_data)
            for obj, rec in zip(new_objects, receptacles):
                object_to_containing_receptacle[obj.handle] = rec
            if sampler_name not in self.episode_data["sampled_objects"]:
                self.episode_data["sampled_objects"][
                    sampler_name
                ] = new_objects
            else:
                # handle duplicate sampler names
                self.episode_data["sampled_objects"][
                    sampler_name
                ] += new_objects
            self.ep_sampled_objects += new_objects
            logger.info(
                f"Sampler {sampler_name} generated {len(new_objects)} new object placements."
            )
            # debug visualization showing each newly added object
            if self._render_debug_obs:
                for new_object in new_objects:
                    self.vdb.look_at(new_object.translation)
                    self.vdb.get_observation()
        t = time.process_time()
        times["t11"] = t - pt
        pt = t


        # simulate the world for a few seconds to validate the placements
        if not self.settle_sim():
            logger.warning(
                "Aborting episode generation due to unstable state."
            )
            return None

        for sampler, target_sampler_info in zip(
            self._target_samplers.values(), self.cfg.object_target_samplers
        ):
            sampler.object_instance_set = [
                x
                for y in target_sampler_info["params"]["object_samplers"]
                for x in self.episode_data["sampled_objects"][y]
            ]
            sampler.object_set = [
                x.creation_attributes.handle
                for x in sampler.object_instance_set
            ]
        t = time.process_time()
        times["t12"] = t - pt
        pt = t


        target_refs: Dict[str, str] = {}

        # sample targets
        handle_to_obj = {obj.handle: obj for obj in self.ep_sampled_objects}
        for sampler_name, target_sampler in self._target_samplers.items():
            obj_sampler_name = targ_sampler_name_to_obj_sampler_names[
                sampler_name
            ][0]
            new_target_objects = target_sampler.sample(
                self.sim,
                recep_tracker,
                snap_down=True,
                vdb=self.vdb,
                target_receptacles=target_receptacles[obj_sampler_name],
                goal_receptacles=goal_receptacles[sampler_name],
                object_to_containing_receptacle=object_to_containing_receptacle,
            )
            if new_target_objects is None:
                times["t13"] = time.process_time() - t
                self.gen_cleanup(times)
                return None
            for target_handle, (
                new_target_obj,
                _,
            ) in new_target_objects.items():
                match_obj = handle_to_obj[target_handle]

                dist = np.linalg.norm(
                    match_obj.translation - new_target_obj.translation
                )
                if dist < self.cfg.min_dist_from_start_to_goal:
                    times["t13"] = time.process_time() - t
                    self.gen_cleanup(times)
                    return None

            # cache transforms and add visualizations
            for i, (instance_handle, value) in enumerate(
                new_target_objects.items()
            ):
                target_object, target_receptacle = value
                target_receptacles[obj_sampler_name][i] = target_receptacle
                assert (
                    instance_handle not in self.episode_data["sampled_targets"]
                ), f"Duplicate target for instance '{instance_handle}'."
                rom = self.sim.get_rigid_object_manager()
                target_bb_size = (
                    target_object.root_scene_node.cumulative_bb.size()
                )
                target_transform = target_object.transformation
                self.episode_data["sampled_targets"][
                    instance_handle
                ] = np.array(target_transform)
                target_refs[
                    instance_handle
                ] = f"{sampler_name}|{len(target_refs)}"
                rom.remove_object_by_handle(target_object.handle)
                if self._render_debug_obs:
                    sutils.add_transformed_wire_box(
                        self.sim,
                        size=target_bb_size / 2.0,
                        transform=target_transform,
                    )
                    self.vdb.look_at(target_transform.translation)
                    self.vdb.debug_line_render.set_line_width(2.0)
                    self.vdb.debug_line_render.draw_transformed_line(
                        target_transform.translation,
                        rom.get_object_by_handle(instance_handle).translation,
                        mn.Color4(1.0, 0.0, 0.0, 1.0),
                        mn.Color4(1.0, 0.0, 0.0, 1.0),
                    )
                    self.vdb.get_observation()
        t = time.process_time()
        times["t13"] = t - pt
        pt = t


        # viewpoints = {}
        # for target_handle, target_transform in self.episode_data["sampled_targets"].items():
        #     # viewpoints.append(self.generate_viewpoints(target_handle, target_transform))
        #     target_viewpoints = self.generate_viewpoints(target_handle, target_transform)
        #     if len(target_viewpoints) == 0:
        #         self.gen_cleanup(times)
        #         return None
        #     viewpoints[target_handle] = target_viewpoints


        # collect final object states and serialize the episode
        # TODO: creating shortened names should be automated and embedded in the objects to be done in a uniform way
        sampled_rigid_object_states = []
        for sampled_obj in self.ep_sampled_objects:
            creation_attrib = sampled_obj.creation_attributes
            file_handle = creation_attrib.handle.split(
                creation_attrib.file_directory
            )[-1].split("/")[-1]
            sampled_rigid_object_states.append(
                (
                    file_handle,
                    np.array(sampled_obj.transformation),
                )
            )
        t = time.process_time()
        times["t14"] = t - pt
        pt = t


        self.num_ep_generated += 1

        def extract_recep_info(recep):
            return (recep.parent_object_handle, recep.parent_link)

        save_target_receps = [
            extract_recep_info(x) for x in all_target_receptacles
        ]
        save_goal_receps = [
            extract_recep_info(x) for x in all_goal_receptacles
        ]

        name_to_receptacle = {
            k: v.name for k, v in object_to_containing_receptacle.items()
        }
        t = time.process_time()
        times["t15"] = t - pt
        pt = t

        ep = RearrangeEpisode(
            scene_dataset_config=self.cfg.dataset_path,
            additional_obj_config_paths=self.cfg.additional_object_paths,
            episode_id=str(self.num_ep_generated - 1),
            start_position=[0, 0, 0],
            start_rotation=[
                0,
                0,
                0,
                1,
            ],
            scene_id=ep_scene_handle,
            ao_states=ao_states,
            rigid_objs=sampled_rigid_object_states,
            targets=self.episode_data["sampled_targets"],
            target_view_locations={},#viewpoints,
            target_receptacles=save_target_receps,
            goal_receptacles=save_goal_receps,
            markers=self.cfg.markers,
            name_to_receptacle=name_to_receptacle,
            info={"object_labels": target_refs},
        )

        t = time.process_time()
        times["t16"] = t - pt
        pt = t

        for k, v in times.items():
            self.times[k] += v
        print(times)
        
        return ep
    
    def generate_viewpoints(self, object_handle: str, object_transform: mn.Matrix4) -> List[np.ndarray]:
        rom = self.sim.get_rigid_object_manager()
        # import pdb
        # pdb.set_trace()
        # obj = rom.add_object_by_template_handle(object_handle)
        obj = rom.get_object_by_handle(object_handle)
        assert obj is not None
        obj.transformation = object_transform
        object_id = obj.object_id
        object_aabb = get_bb(obj)
        object_position = object_aabb.center()
        eps = 1e-5

        # object_nodes = self.sim.get_object_visual_scene_nodes(object_id)
        # assert len(object_nodes) == 1
        # semantic_id = object_nodes[0].semantic_id

        max_distance = 1.0
        cell_size = 0.3/2.0
        x_len, _, z_len = object_aabb.size() / 2.0 + mn.Vector3(max_distance)
        x_bxp = np.arange(-x_len, x_len + eps, step=cell_size) + object_position[0]
        z_bxp = np.arange(-z_len, z_len + eps, step=cell_size) + object_position[2]
        candidate_poses = [
            np.array([x, object_position[1], z])
            for x, z in itertools.product(x_bxp, z_bxp)
        ]

        def down_is_navigable(pt, search_dist=2.0):
            pf = self.sim.pathfinder
            delta_y = 0.05
            max_steps = int(search_dist / delta_y)
            step = 0
            is_navigable = pf.is_navigable(pt, 2)
            while not is_navigable:
                pt[1] -= delta_y
                is_navigable = pf.is_navigable(pt)
                step += 1
                if step == max_steps:
                    return False
            return True

        def _get_iou(x, y, z):
            pt = np.array([x, y, z])

            # TODO: What is this?
            # if not (
            #     object_obb.distance(pt) <= max_distance
            #     and habitat_sim.geo.OBB(object_aabb).distance(pt) <= max_distance
            # ):
            #     return -0.5, pt, None, 'Unknown error'

            if not down_is_navigable(pt):
                return -1.0, pt, None, 'Down is not navigable'
            pf = self.sim.pathfinder
            pt = np.array(pf.snap_point(pt))

            goal_direction = object_position - pt
            goal_direction[1] = 0

            q = direction_to_quaternion(goal_direction)

            cov = 0
            agent = self.sim.get_agent(0)
            for act in [
                "look_down",
                "look_up",
                "look_up",
                # HabitatSimActions.LOOK_DOWN,
                # HabitatSimActions.LOOK_UP,
                # HabitatSimActions.LOOK_UP,
            ]:
                agent.act(act)
                for v in agent._sensors.values():
                    v.set_transformation_from_spec()
                # obs = self.sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
                agent_state = agent.get_state()
                agent_state.position = pt
                agent_state.rotation = q
                agent.set_state(agent_state, reset_sensors=False)
                obs = self.sim.get_sensor_observations(0)
                import pdb
                pdb.set_trace()
                cov += compute_pixel_coverage(obs["semantic"], object_id)
                # cov += semantic_id in obs["semantic"]

            # from habitat.utils.visualizations.utils import observations_to_image
            # import imageio
            # import os
            # import cv2

            # obs = self.sim.get_observations_at(pt, q, keep_agent_at_new_pose=True)
            # rgb_obs = np.ascontiguousarray(obs['rgb'][..., :3])
            # sem_obs = (obs["semantic"] == object_id).astype(np.uint8) * 255
            # contours, _ = cv2.findContours(sem_obs, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # rgb_obs = cv2.drawContours(rgb_obs, contours, -1, (0, 255, 0), 4)
            # imageio.imsave(
            #     os.path.join(
            #         "data/images/objnav_dataset_gen",
            #         f"{object_category_name}_{object_name_id}_{object_id}_{x}_{z}_.png",
            #     ), rgb_obs
            # )

            return cov, pt, q, 'Success'
        
        candidate_poses_ious_orig = list(_get_iou(*pos) for pos in candidate_poses)
        n_unknown_rejected = 0
        n_down_not_navigable_rejected = 0
        for p in candidate_poses_ious_orig:
            if p[-1] == 'Unknown error':
                n_unknown_rejected += 1
            elif p[-1] == 'Down is not navigable':
                n_down_not_navigable_rejected += 1
        candidate_poses_ious_orig_2 = [
            p for p in candidate_poses_ious_orig
            if p[0] > 0
        ]

        # Reject candidate_poses that do not satisfy island radius constraints
        candidate_poses_ious = [
            p for p in candidate_poses_ious_orig_2
            if self.sim.pathfinder.island_radius(p[1]) >= ISLAND_RADIUS_LIMIT
        ]

        keep_thresh = 0
        view_locations = [
            ObjectViewLocation(
                AgentState(pt.tolist(), quat_to_coeffs(q).tolist()), iou
            )
            for iou, pt, q, _ in candidate_poses_ious
            if iou is not None and iou > keep_thresh
        ]
        view_locations = sorted(view_locations, reverse=True, key=lambda v: v.iou)
        # view_locations = [view.agent_state.position for view in view_locations]

        return view_locations

    def initialize_sim(self, scene_name: str, dataset_path: str) -> None:
        """
        Initialize a new Simulator object with a selected scene and dataset.
        """
        # Setup a camera coincident with the agent body node.
        # For debugging visualizations place the default agent where you want the camera with local -Z oriented toward the point of focus.
        import time
        t = time.process_time()
        camera_resolution = [540, 720]
        sensors = {
            "rgb": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            },
            "semantic": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [256, 256],
                "position": [0, 0, 0],
                "orientation": [0, 0, 0.0],
            },
        }

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_dataset_config_file = dataset_path
        backend_cfg.scene_id = scene_name
        backend_cfg.enable_physics = True
        if not self._render_debug_obs:
            # don't bother loading textures if not intending to visualize the generation process
            backend_cfg.create_renderer = False

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            # sensor_spec = habitat_sim.EquirectangularSensorSpec()
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_spec.sensor_subtype = (
                habitat_sim.SensorSubType.EQUIRECTANGULAR
            )
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            "turn_left": ActionSpec("turn_left", ActuationSpec(amount=10.0)),
            "turn_right": ActionSpec("turn_right", ActuationSpec(amount=10.0)),
            "look_up": ActionSpec("look_up", ActuationSpec(amount=10.0)),
            "look_down": ActionSpec("look_down", ActuationSpec(amount=10.0)),
        }

        hab_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        if self.sim is None:
            self.sim = habitat_sim.Simulator(hab_cfg)

            object_attr_mgr = self.sim.get_object_template_manager()
            for object_path in self.cfg.additional_object_paths:
                object_attr_mgr.load_configs(osp.abspath(object_path))
        else:
            if self.sim.config.sim_cfg.scene_id == scene_name:
                # # we need to force a reset, so reload the NONE scene
                # # TODO: we should fix this to provide an appropriate reset method
                # proxy_backend_cfg = habitat_sim.SimulatorConfiguration()
                # proxy_backend_cfg.scene_id = "NONE"
                # proxy_hab_cfg = habitat_sim.Configuration(
                #     proxy_backend_cfg, [agent_cfg]
                # )
                # self.sim.reconfigure(proxy_hab_cfg)
                rom = self.sim.get_rigid_object_manager()
                for object in rom.get_object_handles():
                    rom.remove_object_by_handle(object)
                aom = self.sim.get_articulated_object_manager()
                for ao_handle in self.ao_states.keys():
                    aom.get_object_by_handle(ao_handle).clear_joint_states()
                try:
                    assert len(rom.get_object_handles()) == 0
                except:
                    import pdb
                    pdb.set_trace()
            else:
                self.sim.reconfigure(hab_cfg)
        self.ep_sampled_objects = []
        print("sim setup", time.process_time() - t)
        t = time.process_time()

        # setup the debug camera state to the center of the scene bounding box
        scene_bb = (
            self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        )
        self.sim.agents[0].scene_node.translation = scene_bb.center()

        # initialize the debug visualizer
        self.vdb = DebugVisualizer(
            self.sim, output_path="rearrange_ep_gen_output/"
        )
        print("debug viz", time.process_time() - t)

    def settle_sim(
        self, duration: float = 5.0, make_video: bool = True
    ) -> bool:
        """
        Run dynamics for a few seconds to check for stability of newly placed objects and optionally produce a video.
        Returns whether or not the simulation was stable.
        """
        if len(self.ep_sampled_objects) == 0:
            return True
        # assert len(self.ep_sampled_objects) > 0

        scene_bb = (
            self.sim.get_active_scene_graph().get_root_node().cumulative_bb
        )
        new_obj_centroid = mn.Vector3()
        spawn_positions = {}
        for new_object in self.ep_sampled_objects:
            spawn_positions[new_object.handle] = new_object.translation
            new_obj_centroid += new_object.translation
        new_obj_centroid /= len(self.ep_sampled_objects)
        settle_db_obs: List[Any] = []
        if self._render_debug_obs:
            self.vdb.get_observation(
                look_at=new_obj_centroid,
                look_from=scene_bb.center(),
                obs_cache=settle_db_obs,
            )

        while self.sim.get_world_time() < duration:
            self.sim.step_world(1.0 / 30.0)
            if self._render_debug_obs:
                self.vdb.get_observation(obs_cache=settle_db_obs)

        # check stability of placements
        logger.info("Computing placement stability report:")
        max_settle_displacement = 0
        error_eps = 0.1
        unstable_placements = []
        for new_object in self.ep_sampled_objects:
            error = (
                spawn_positions[new_object.handle] - new_object.translation
            ).length()
            max_settle_displacement = max(max_settle_displacement, error)
            if error > error_eps:
                unstable_placements.append(new_object.handle)
                logger.info(
                    f"    Object '{new_object.handle}' unstable. Moved {error} units from placement."
                )
        logger.info(
            f" : unstable={len(unstable_placements)}|{len(self.ep_sampled_objects)} ({len(unstable_placements)/len(self.ep_sampled_objects)*100}%) : {unstable_placements}."
        )
        logger.info(
            f" : Maximum displacement from settling = {max_settle_displacement}"
        )
        # TODO: maybe draw/display trajectory tubes for the displacements?

        if self._render_debug_obs and make_video:
            self.vdb.make_debug_video(
                prefix="settle_", fps=30, obs_cache=settle_db_obs
            )

        # return success or failure
        return len(unstable_placements) == 0
