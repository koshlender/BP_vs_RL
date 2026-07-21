#!/usr/bin/env python
from pathlib import Path
import csv, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config,write_json
from src.environment.simulated_env import QueueNetworkEnv
from src.agents.q_learning import TabularQLearningAgent
from src.rewards.qplf import backpressure_weights, cyclic_green_times

def run_policy(name, agent=None):
    cfg=load_config('configs/chapter_4_5.json'); env=QueueNetworkEnv(duration=120, seed=cfg['seed']); s=env.reset(); done=False; reward=0
    while not done:
        if name=='backpressure': a=int(backpressure_weights(s,[0,0,0,0], [[1,1,0,0],[0,0,1,1]])[1] > backpressure_weights(s,[0,0,0,0], [[1,1,0,0],[0,0,1,1]])[0])
        elif agent: a=agent.act(s, evaluate=True)
        else: a=0
        s,r,done,_=env.step(a); reward+=r
    m=env.metrics(); m.update({'policy':name,'reward':reward}); return m

def main():
    agent=TabularQLearningAgent.load('results/checkpoints/short_q_agent.pkl') if Path('results/checkpoints/short_q_agent.pkl').exists() else None
    rows=[run_policy('semi_coordinated_rl_short', agent), run_policy('independent_fixed'), run_policy('backpressure')]
    Path('results/raw').mkdir(parents=True, exist_ok=True)
    with open('results/raw/short_evaluation.csv','w',newline='') as f: w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    write_json('results/raw/chapter5_comparison.json', {'thesis_table_5_2_seconds': load_config('configs/chapter_4_5.json')['thesis_reported'], 'reproduced_short_run': rows})
if __name__=='__main__': main()
