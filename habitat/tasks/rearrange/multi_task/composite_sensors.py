#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    ObjectToGoalDistance,
    RearrangeReward,
)
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_measure
class MoveObjectsReward(RearrangeReward):
    """
    A reward based on L2 distances to object/goal.
    """

    cls_uuid: str = "move_obj_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MoveObjectsReward.cls_uuid

    def __init__(self, *args, **kwargs):
        self._cur_rearrange_step = 0
        super().__init__(*args, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._cur_rearrange_step = 0
        self._prev_holding_obj = False
        self._did_give_pick_reward = {}
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
                EndEffectorToObjectDistance.cls_uuid,
            ],
        )

        to_goal = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        to_obj = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()
        self._prev_measures = (to_obj, to_goal)

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        idxs, _ = self._sim.get_targets()
        targ_obj_idx = idxs[self._cur_rearrange_step]
        abs_targ_obj_idx = self._sim.scene_obj_ids[targ_obj_idx]
        targ_obj_idx = str(targ_obj_idx)
        num_targs = len(idxs)

        to_goal = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()
        to_obj = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()

        is_holding_obj = self._sim.grasp_mgr.snap_idx == abs_targ_obj_idx
        if is_holding_obj:
            dist = to_goal[targ_obj_idx]
            dist_diff = (
                self._prev_measures[1][targ_obj_idx] - to_goal[targ_obj_idx]
            )
        else:
            dist = to_obj[targ_obj_idx]
            dist_diff = (
                self._prev_measures[0][targ_obj_idx] - to_obj[targ_obj_idx]
            )

        if (
            is_holding_obj
            and not self._prev_holding_obj
            and self._cur_rearrange_step not in self._did_give_pick_reward
        ):
            self._metric += self._config.PICK_REWARD
            self._did_give_pick_reward[self._cur_rearrange_step] = True

        if (
            dist < self._config.SUCCESS_DIST
            and not is_holding_obj
            and self._cur_rearrange_step < num_targs
        ):
            self._metric += self._config.SINGLE_REARRANGE_REWARD
            self._cur_rearrange_step += 1
            self._cur_rearrange_step = min(
                self._cur_rearrange_step, num_targs - 1
            )

        self._metric += self._config.DIST_REWARD * dist_diff
        self._prev_measures = (to_obj, to_goal)
        self._prev_holding_obj = is_holding_obj


@registry.register_measure
class CompositeSparseReward(Measure):
    cls_uuid: str = "composite_sparse_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeSparseReward.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self._config.SLACK_REWARD

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config


@registry.register_measure
class CompositeReward(Measure):
    """
    The reward based on where the agent currently is in the hand defined solution list.
    """

    cls_uuid: str = "composite_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeReward.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._prev_node_idx = None

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [CompositeNodeIdx.cls_uuid],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0.0
        node_measure = task.measurements.measures[CompositeNodeIdx.cls_uuid]

        node_idx = node_measure.get_metric()["node_idx"]
        if self._prev_node_idx is None:
            self._prev_node_idx = node_idx
        elif node_idx > self._prev_node_idx:
            self._metric += self._config.STAGE_COMPLETE_REWARD

        cur_task_cfg = task.get_inferrred_node_task()._config

        if "REWARD_MEASURE" not in cur_task_cfg:
            raise ValueError(
                f"Cannot find REWARD_MEASURE key in {list(cur_task_cfg.keys())}"
            )
        cur_task_reward = task.measurements.measures[
            cur_task_cfg.REWARD_MEASURE
        ].get_metric()
        self._metric += cur_task_reward

        self._prev_node_idx = node_idx


@registry.register_measure
class DoesWantTerminate(Measure):
    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["REARRANGE_STOP"].does_want_terminate


@registry.register_measure
class CompositeBadCalledTerminate(Measure):
    cls_uuid: str = "composite_bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeBadCalledTerminate.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, CompositeSuccess.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            CompositeSuccess.cls_uuid
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_measure
class CompositeSuccess(Measure):
    """
    Did satisfy all the goal predicates?
    """

    cls_uuid: str = "composite_success"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeSuccess.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        self._metric = task.pddl_problem.is_expr_true(
            task.pddl_problem.goal
        ) and (does_action_want_stop or not self._config.MUST_CALL_STOP)
        if does_action_want_stop:
            task.should_end = True


@registry.register_measure
class CompositeStageGoals(Measure):
    """
    Adds to the metrics `[TASK_NAME]_success`: Did the agent complete a
        particular stage defined in `stage_goals`.
    """

    _stage_succ: List[str]
    cls_uuid: str = "composite_stage_goals"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeStageGoals.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._stage_succ = []
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        self._metric = {}
        for stage_name, logical_expr in task.pddl_problem.stage_goals.items():
            succ_k = f"{stage_name}_success"
            if stage_name in self._stage_succ:
                self._metric[succ_k] = 1.0
            else:
                if task.pddl_problem.is_expr_true(logical_expr):
                    self._metric[succ_k] = 1.0
                    self._stage_succ.append(stage_name)
                else:
                    self._metric[succ_k] = 0.0


@registry.register_measure
class CompositeNodeIdx(Measure):
    """
    Adds several keys to the metrics dictionary:
        - `reached_i`: Did the agent succeed in sub-task at index `i` of the
          sub-task `solution` list?
        - `node_idx`: Index of the agent in completing the sub-tasks from
          the `solution` list.
    """

    cls_uuid: str = "composite_node_idx"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._stage_succ = []

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return CompositeNodeIdx.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = {}
        inf_cur_task_cfg = task.get_inferrred_node_task()._config
        if "SUCCESS_MEASURE" not in inf_cur_task_cfg:
            raise ValueError(
                f"SUCCESS_MEASURE key not found in config: {inf_cur_task_cfg}"
            )

        is_succ = task.measurements.measures[
            inf_cur_task_cfg.SUCCESS_MEASURE
        ].get_metric()
        if is_succ:
            task.increment_inferred_solution_idx(episode)
            rearrange_logger.debug(
                f"Completed {inf_cur_task_cfg.TYPE}, incremented node to {task.get_inferrred_node_task()}"
            )

        node_idx = task.get_inferred_node_idx()
        for i in range(task.num_solution_subtasks):
            self._metric[f"reached_{i}"] = task.get_inferred_node_idx() >= i
        self._metric["node_idx"] = node_idx
