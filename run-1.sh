uv run  main.py --env_name=cube-single-play-oraclerep-v0 \
    --dataset_dir=cube-single-play-v0/ --dataset_replace_interval=1000 \
    --agent=agents/dsharsa.py --agent.q_agg=min --agent.subgoal_steps=50 --agent.actor_p_trajgoal=1.0 \
    --agent.actor_p_randomgoal=0.0 --agent.actor_geom_sample=True
