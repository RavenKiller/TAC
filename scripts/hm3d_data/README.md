1. Require habitat 2.4

2. Require hack several parts of habitat-lab

    1. habitat-lab/habitat/tasks/nav/nav.py: SPL.update_metric
        
        solving divide by zero error

    ```
    +++ 
    @@ -598,19 +598,26 @@
            current_position = self._sim.get_agent_state().position
            self._agent_episode_distance += self._euclidean_distance(
                current_position, self._previous_position
            )
    
            self._previous_position = current_position
    -
    -        self._metric = ep_success * (
    -            self._start_end_episode_distance
    -            / max(
    +        dist = max(
                    self._start_end_episode_distance, self._agent_episode_distance
                )
    -        )
    +        if dist==0:
    +            dist_ratio = 1
    +        else:
    +            dist_ratio = self._start_end_episode_distance / dist
    +        self._metric = ep_success * dist_ratio
    +        # self._metric = ep_success * (
    +        #     self._start_end_episode_distance
    +        #     / max(
    +        #         self._start_end_episode_distance, self._agent_episode_distance
    +        #     )
    +        # )
    
    
    @registry.register_measure
    class SoftSPL(SPL):
        r"""Soft SPL
    ```

    2. habitat-lab/habitat/core/env.py: Env.\_\_init\_\_
        
        adding index to select part of episodes
    ```
    @@ -65,13 +65,13 @@
        _episode_start_time: Optional[float]
        _episode_over: bool
        _episode_from_iter_on_reset: bool
        _episode_force_changed: bool
    
        def __init__(
    -        self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None
    +        self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None, index_start=-1, index_stop=-1, 
        ) -> None:
            """Constructor
    
            :param config: config for the environment. Should contain id for
                simulator and ``task_name`` which are passed into ``make_sim`` and
                ``make_task``.
    @@ -85,12 +85,16 @@
            self._config = config
            self._dataset = dataset
            if self._dataset is None and config.dataset.type:
                self._dataset = make_dataset(
                    id_dataset=config.dataset.type, config=config.dataset
                )
    +            # choose limited episodes
    +            if index_start!=-1 and index_stop!=-1:
    +                random.shuffle(self._dataset.episodes)
    +                self._dataset.episodes = self._dataset.episodes[index_start:index_stop]
    
            self._current_episode = None
            self._episode_iterator = None
            self._episode_from_iter_on_reset = True
            self._episode_force_changed = False
    ```
3. Run the data collecting script:
    1. Manually start multi-processing: `sh hm3d_data_mp.sh run` to start; `sh hm3d_data_mp.sh kill` to interrupt.
    2. Automatically start multi-processing: `python hm3d_data_mp.py --world_size 4`. The `--total_episodes` param can be used to specify how many episodes are collected.