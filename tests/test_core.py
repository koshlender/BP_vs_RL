from pathlib import Path
import subprocess, sys, math
from src.utils.config import load_config
from src.state.state_builder import StateBuilder, StateSpec
from src.environment.actions import PhasePlan, action_to_phase, transition_phase
from src.rewards.qlf import queue_length_function, reward_from_qlf
from src.rewards.qplf import queue_pressure_lyapunov_function, backpressure_weights, cyclic_green_times
from src.metrics.stops import StopCounter
from src.metrics.platoon import PlatoonProgressionTracker
from src.agents.q_learning import TabularQLearningAgent

def test_config_loads(): assert load_config('configs/chapter_4_5.json')['seed']==7

def test_state_builder():
    b=StateBuilder(StateSpec(('N','S'),('E',),('J0',), max_queue=10)); s=b.build({'N':5,'S':20,'E':1},{'J0':1})
    assert b.dimension==4 and s==[.5,1,.1,1]

def test_actions():
    p=PhasePlan({0:0,1:2},{(0,1):1,(1,0):3}); assert action_to_phase(1,p)==2; assert transition_phase(0,1,p)==(1,2)

def test_qlf_qplf():
    assert queue_length_function([0,0])==0; assert reward_from_qlf([2,0],[1,0])==3
    assert queue_pressure_lyapunov_function([5,1],[2,2])==9
    assert backpressure_weights([5,1],[2,2],[[1,0],[0,1]])==[3,-1]
    assert math.isclose(sum(cyclic_green_times([1,2],60,6,.1)),54)

def test_stop_counter_cases():
    c=StopCounter(minimum_stop_duration=2)
    c.update('v',5); c.update('v',0); c.update('v',0); assert c.summary()['total_stop_events']==1
    c.update('v',0); c.update('v',2); c.update('v',0); c.update('v',0); c.mark_completed('v')
    assert c.summary()['total_stop_events']==2 and c.summary()['completed_vehicles']==1

def test_platoon_cases():
    p=PlatoonProgressionTracker({'r'}); p.cross_first('v','r',0); p.reach_second('v',10); p.clear_second('v',20)
    assert p.summary()['platoon_progression_percentage']==100
    p.cross_first('w','r',0); p.observe_corridor('w',0); p.reach_second('w',10); p.clear_second('w',20)
    assert p.summary()['vehicles_progressing_without_stopping']==1
    assert PlatoonProgressionTracker({'x'}).summary()['platoon_progression_percentage'] is None

def test_checkpoint(tmp_path):
    a=TabularQLearningAgent(); a.update([1,2],0,1,[0,0],True); f=tmp_path/'a.pkl'; a.save(f); assert TabularQLearningAgent.load(f).q

def test_end_to_end_scripts():
    subprocess.check_call([sys.executable,'scripts/train.py']); subprocess.check_call([sys.executable,'scripts/evaluate.py']); subprocess.check_call([sys.executable,'scripts/plot_results.py'])
    assert Path('results/raw/short_evaluation.csv').exists() and Path('plots/short_training_reward.svg').exists()

def test_piecewise_linear_agent_and_qplf_script(tmp_path):
    from src.agents.pwl_q_learning import PiecewiseLinearQAgent
    agent = PiecewiseLinearQAgent(epsilon=0.0)
    state = [1.0, 3.0]
    feats0 = agent.features(state, 0)
    feats1 = agent.features(state, 1)
    assert len(feats0) == 8 and feats0[:4] != [0.0] * 4 and feats0[4:] == [0.0] * 4
    assert feats1[:4] == [0.0] * 4 and feats1[4:] != [0.0] * 4
    assert sum(abs(x) for x in feats0) <= 1.0
    action = agent.act(state, evaluate=True)
    agent.update(state, action, 1.0, [0.5, 2.0], False)
    assert agent.q_value(state, action) != 0.0
    subprocess.check_call([sys.executable, 'scripts/run_qplf_experiments.py'])
    assert Path('results/raw/qplf_summary.csv').exists()

def test_real_sumo_availability_check_reports_status():
    from src.environment.sumo_env import check_sumo_availability
    availability = check_sumo_availability()
    assert isinstance(availability.traci_importable, bool)
    assert isinstance(availability.sumolib_importable, bool)
    assert isinstance(availability.errors, list)
