from pathlib import Path
import subprocess, sys, math
from src.utils.config import load_config
from src.state.state_builder import StateBuilder, StateSpec, chapter4_quantize_queue, chapter4_local_state, chapter4_queue_label, chapter4_category_label
from src.environment.actions import PhasePlan, action_to_phase, transition_phase, chapter4_green_time_actions, chapter4_cooperative_actions, CH4_PHASE_SEQUENCES
from src.rewards.qlf import queue_length_function, reward_from_qlf
from src.rewards.chapter4 import chapter4_queue_reward, chapter4_queue_reward_result
from src.rewards.qplf import queue_pressure_lyapunov_function, backpressure_weights, cyclic_green_times
from src.metrics.stops import StopCounter
from src.metrics.platoon import PlatoonProgressionTracker
from src.agents.q_learning import TabularQLearningAgent
from src.thesis.chapter_constants import THESIS_CYCLE_TIME_SECONDS, THESIS_TOTAL_YELLOW_TIME_SECONDS, THESIS_AVAILABLE_GREEN_TIME_SECONDS, CH4_QUEUE_CATEGORIES

def test_config_loads(): assert load_config('configs/chapter_4_5.json')['seed']==7

def test_state_builder():
    b=StateBuilder(StateSpec(('N','S'),('E',),('J0',))); s=b.build({'N':5,'S':20,'E':21},{'J0':16})
    assert b.dimension==4 and s==[0,1,2,16]


def test_chapter4_state_action_reward_equations():
    boundary_values = [0, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
    expected_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    expected_labels = ['k', 'k', 'l', 'l', 'm', 'm', 'n', 'n', 'o', 'o', 'p']
    assert [chapter4_quantize_queue(q) for q in boundary_values] == expected_ids
    assert [chapter4_queue_label(q) for q in boundary_values] == expected_labels
    assert [chapter4_category_label(i) for i in range(6)] == ['k', 'l', 'm', 'n', 'o', 'p']
    for q in range(0, 200):
        matches = [c for c in CH4_QUEUE_CATEGORIES if q >= c.lower_inclusive and (c.upper_inclusive is None or q <= c.upper_inclusive)]
        assert len(matches) == 1
    assert chapter4_local_state([0, 12, 35, 99]) == (0, 1, 3, 5)
    green_actions = chapter4_green_time_actions()
    assert THESIS_CYCLE_TIME_SECONDS == 80 and THESIS_TOTAL_YELLOW_TIME_SECONDS == 16
    assert green_actions and all(len(a) == 4 for a in green_actions)
    assert all(sum(a) == THESIS_AVAILABLE_GREEN_TIME_SECONDS for a in green_actions)
    assert all(all(g in {4,8,12,16,20,24,28,32} for g in a) for a in green_actions)
    assert len(CH4_PHASE_SEQUENCES) == 8
    cooperative = chapter4_cooperative_actions()
    assert len(cooperative) == len(green_actions) * 8
    result = chapter4_queue_reward_result([2, 3, 5, 10])
    assert result.reward == 0.05 and result.used_thesis_equation and not result.zero_queue_policy_applied
    one_vehicle = chapter4_queue_reward_result([0, 0, 1, 0])
    assert one_vehicle.reward == 1.0 and one_vehicle.used_thesis_equation
    zero = chapter4_queue_reward_result([0, 0, 0, 0])
    assert zero.reward == 0.0 and not zero.used_thesis_equation and zero.zero_queue_policy_applied
    zero_reward = chapter4_queue_reward([0, 0, 0, 0])
    td_target = zero_reward + 0.95 * 0.0
    assert math.isfinite(zero_reward) and math.isfinite(td_target)

def test_actions():
    p=PhasePlan({0:0,1:2},{(0,1):1,(1,0):3}); assert action_to_phase(1,p)==2; assert transition_phase(0,1,p)==(1,2)

def test_qlf_qplf():
    assert queue_length_function([0,0])==0; assert reward_from_qlf([2,0],[1,0])==1
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
    a=TabularQLearningAgent(); action=a.actions[0]; a.update([1,2,3,4],action,1,[0,0,0,0],True); f=tmp_path/'a.pkl'; a.save(f); assert TabularQLearningAgent.load(f).q

def test_end_to_end_scripts():
    subprocess.check_call([sys.executable,'scripts/train.py']); subprocess.check_call([sys.executable,'scripts/evaluate.py']); subprocess.check_call([sys.executable,'scripts/plot_results.py'])
    assert Path('results/raw/short_evaluation.csv').exists() and Path('plots/short_training_reward.svg').exists()

def test_piecewise_linear_agent_and_qplf_script(tmp_path):
    from src.agents.pwl_q_learning import PiecewiseLinearQAgent
    agent = PiecewiseLinearQAgent(epsilon=0.0)
    state = [1.0, 30.0, 51.0, 0.0]
    a0, a1 = agent.actions[0], agent.actions[1]
    feats0 = agent.features(state, a0)
    feats1 = agent.features(state, a1)
    assert len(feats0) == 4 * len(agent.actions) and feats0[:4] != [0.0] * 4 and feats0[4:] == [0.0] * (len(feats0) - 4)
    assert feats1[:4] == [0.0] * 4 and feats1[4:8] != [0.0] * 4
    assert sum(abs(x) for x in feats0) <= 1.0
    action = agent.act(state, evaluate=True)
    agent.update(state, action, 1.0, [0.5, 2.0, 3.0, 4.0], False)
    assert agent.q_value(state, action) != 0.0
    subprocess.check_call([sys.executable, 'scripts/run_qplf_experiments.py'])
    assert Path('results/raw/qplf_summary.csv').exists()

def test_real_sumo_availability_check_reports_status():
    from src.environment.sumo_env import check_sumo_availability
    availability = check_sumo_availability()
    assert isinstance(availability.traci_importable, bool)
    assert isinstance(availability.sumolib_importable, bool)
    assert isinstance(availability.errors, list)


def test_generate_thesis_sumo_assets():
    subprocess.check_call([sys.executable, 'scripts/generate_thesis_sumo.py'])
    assert Path('sumo/thesis_ch4_ch5/two_intersection/two_intersection.rou.xml').exists()
    nine = Path('sumo/thesis_ch4_ch5/nine_intersection/scenario1.rou.xml').read_text()
    assert 'period="40"' in nine and 'period="700"' in nine and 'W3_E3' in nine

def test_eta_sweep_script_outputs_best_eta():
    subprocess.check_call([sys.executable, 'scripts/run_eta_sweep.py'])
    assert Path('results/raw/eta_sweep_best.csv').exists()
