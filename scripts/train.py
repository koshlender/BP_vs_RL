#!/usr/bin/env python
from pathlib import Path
import csv, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config,set_seed,write_json
from src.environment.simulated_env import QueueNetworkEnv
from src.agents.q_learning import TabularQLearningAgent

def main():
    cfg=load_config('configs/chapter_4_5.json'); set_seed(cfg['seed'])
    env=QueueNetworkEnv(duration=int(cfg['simulation']['duration_seconds']), seed=cfg['seed'])
    agent=TabularQLearningAgent(alpha=cfg['rl']['learning_rate'], gamma=cfg['rl']['gamma'], epsilon=cfg['rl']['epsilon_start'])
    rows=[]
    for ep in range(int(cfg['rl']['episodes'])):
        s=env.reset(); total=0; done=False
        while not done:
            a=agent.act(s); ns,r,done,info=env.step(a); agent.update(s,a,r,ns,done); s=ns; total+=r
        rows.append({'episode':ep,'reward':total, **env.metrics()}); agent.epsilon=max(cfg['rl']['epsilon_end'], agent.epsilon*cfg['rl']['epsilon_decay'])
    Path('results/raw').mkdir(parents=True, exist_ok=True)
    with open('results/raw/short_training.csv','w',newline='') as f: w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True); agent.save('results/checkpoints/short_q_agent.pkl')
    write_json('results/raw/short_training_metadata.json', {'config':'configs/chapter_4_5.json','episodes':len(rows),'note':'short cloud run; not thesis-scale 200 episodes'})
if __name__=='__main__': main()
