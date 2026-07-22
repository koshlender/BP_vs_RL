from pathlib import Path
import subprocess, sys, math
from src.utils.config import load_config
from src.state.state_builder import StateBuilder, StateSpec, chapter4_quantize_queue, chapter4_local_state, chapter4_queue_label, chapter4_category_label
from src.environment.actions import PhasePlan, action_to_phase, transition_phase, chapter4_green_time_actions, chapter4_cooperative_actions, CH4_PHASE_SEQUENCES
from src.rewards.qlf import queue_length_function, reward_from_qlf
from src.rewards.chapter4 import chapter4_queue_reward, chapter4_queue_reward_result
from src.rewards.qplf import queue_pressure_lyapunov_function, backpressure_weights, cyclic_green_times
from src.rewards.backpressure import queue_dynamics, phase_weight, cyclic_backpressure_decision, cyclic_backpressure_proportions, deterministic_integer_durations
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
    assert chapter4_queue_reward_result([0, 0, 0, 0], zero_queue_policy="epsilon", epsilon=2).reward == 0.5
    zero_reward = chapter4_queue_reward([0, 0, 0, 0])
    td_target = zero_reward + 0.95 * 0.0
    assert math.isfinite(zero_reward) and math.isfinite(td_target)

def test_actions():
    p=PhasePlan({0:0,1:2},{(0,1):1,(1,0):3}); assert action_to_phase(1,p)==2; assert transition_phase(0,1,p)==(1,2)

def test_qlf_qplf():
    assert queue_length_function([0,0])==0; assert reward_from_qlf([2,0],[1,0])==1
    assert queue_pressure_lyapunov_function([5,1],[2,2])==9
    assert backpressure_weights([5,1],[2,2],[[1,0],[0,1]])==[3,-1]
    assert math.isclose(sum(cyclic_green_times([1,2],80,16,.1)),64)

def test_chapter5_cyclic_backpressure_equations():
    turns = [[0.8, 0.2], [0.1, 0.9]]
    assert queue_dynamics([10, 5], [2, 3], turns) == [10 - 2 + 0.8 * 2 + 0.1 * 3, 5 - 3 + 0.2 * 2 + 0.9 * 3]
    weight = phase_weight([1, 0], [10, 4], [3, 2], turns)
    assert math.isclose(weight, 1 * (10 - (0.8 * 3 + 0.2 * 2)))
    decision = cyclic_backpressure_decision([[1, 0], [0, 1]], [10, 4], [3, 2], turns, eta=0.1, available_green_time=64)
    assert len(decision.weights) == 2 and decision.max_pressure_phase == 0
    assert math.isclose(sum(decision.proportions), 1.0)
    assert math.isclose(sum(decision.green_times), 64.0)
    assert sum(decision.executable_green_times) == 64
    assert deterministic_integer_durations([10.2, 10.2, 10.2, 33.4], 64) == [10, 10, 10, 34]
    for weights in ([5, 5, 5, 5], [10000, 10001, 9999], [-10000, -10001, -9999]):
        proportions = cyclic_backpressure_proportions(weights, eta=1.0)
        assert all(math.isfinite(p) for p in proportions)
        assert math.isclose(sum(proportions), 1.0)

def test_chapter5_backpressure_schedule_executes_all_phases():
    from src.environment.simulated_env import QueueNetworkEnv
    env = QueueNetworkEnv(duration=100, seed=1, demand=[0,0,0,0])
    env.reset(); env.queues = [2.0, 2.0, 2.0, 2.0]
    phases = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    queues, reward, done, info = env.step_phase_schedule(phases, [1,1,1,1], yellow_duration_per_phase=1)
    assert queues == [1.0, 1.0, 1.0, 1.0]
    assert info['departed'] == 4
    assert env.t == 8

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
    summary = Path('results/raw/qplf_summary.csv').read_text()
    assert 'full_state_independent_rl' in summary
    assert 'independent_pwl_qplf' in summary
    assert 'semi_coordinated_pwl_qplf' in summary
    assert 'centralized_full_state_rl' in summary
    assert 'cyclic_queue_backpressure' in summary


def test_generate_grid_network_uses_grid_length(monkeypatch, tmp_path):
    from src.environment import sumo_env
    monkeypatch.setattr(sumo_env, "require_sumo", lambda: sumo_env.SumoAvailability("sumo", "netgenerate", True, True, []))
    calls = []
    def fake_check_call(cmd):
        calls.append(cmd)
    monkeypatch.setattr(sumo_env.subprocess, "check_call", fake_check_call)
    sumo_env.generate_grid_network(tmp_path / "grid.net.xml")
    assert calls and "--grid.length" in calls[0]
    assert "--default.length" not in calls[0]



def test_generate_grid_network_accepts_colab_sigsegv_when_output_exists(monkeypatch, tmp_path):
    from src.environment import sumo_env
    monkeypatch.setattr(sumo_env, "require_sumo", lambda: sumo_env.SumoAvailability("sumo", "netgenerate", True, True, []))
    def fake_check_call(cmd):
        (tmp_path / "grid.net.xml").write_text("<net/>\n", encoding="utf-8")
        raise sumo_env.subprocess.CalledProcessError(-11, cmd)
    monkeypatch.setattr(sumo_env.subprocess, "check_call", fake_check_call)
    sumo_env.generate_grid_network(tmp_path / "grid.net.xml")

def test_generate_grid_network_falls_back_to_netconvert(monkeypatch, tmp_path):
    from src.environment import sumo_env
    monkeypatch.setattr(sumo_env, "require_sumo", lambda: sumo_env.SumoAvailability("sumo", "netgenerate", True, True, []))
    monkeypatch.setattr(sumo_env.shutil, "which", lambda name: "netconvert" if name == "netconvert" else None)
    calls = []
    def fake_check_call(cmd):
        calls.append(cmd)
        if cmd[0] == "netgenerate":
            raise sumo_env.subprocess.CalledProcessError(-11, cmd)
    monkeypatch.setattr(sumo_env.subprocess, "check_call", fake_check_call)
    sumo_env.generate_grid_network(tmp_path / "grid.net.xml")
    assert len(calls) == 3
    assert calls[-1][0] == "netconvert"
    assert (tmp_path / "grid.nod.xml").exists()
    assert (tmp_path / "grid.edg.xml").exists()

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

def test_thesis_sumo_metrics_are_thesis_named(tmp_path):
    from scripts.run_thesis_sumo_experiments import summarize_tripinfo, SCENARIOS, CONTROL_ALGORITHM
    tripinfo = tmp_path / "thesis_sample_tripinfo.xml"
    tripinfo.write_text(
        '<tripinfos>\n'
        '  <tripinfo id="v0" duration="10.0" waitingTime="2.0" waitingCount="1"/>\n'
        '  <tripinfo id="v1" duration="30.0" waitingTime="4.0" waitingCount="3"/>\n'
        '</tripinfos>\n',
        encoding="utf-8",
    )
    metrics = summarize_tripinfo(tripinfo)
    assert metrics["completed_vehicles"] == 2
    assert metrics["mean_travel_time_seconds"] == 20.0
    assert metrics["mean_waiting_time_seconds"] == 3.0
    assert metrics["total_stop_events"] == 4
    assert metrics["average_stops_per_completed_vehicle"] == 2.0
    assert all(str(item["scenario"]).startswith(("chapter4", "chapter5")) for item in SCENARIOS)
    assert CONTROL_ALGORITHM == "sumo_static_tls_baseline"

def test_thesis_algorithm_comparison_outputs_requested_artifacts():
    subprocess.check_call([sys.executable, 'scripts/run_thesis_algorithm_comparison.py', '--episodes', '2', '--duration', '80'])
    expected_raw = [
        'results/raw/thesis_algorithm_episode_delay.csv',
        'results/raw/thesis_eta_vs_delay.csv',
        'results/raw/thesis_queue_vs_time.csv',
        'results/raw/thesis_algorithm_summary.csv',
        'results/raw/thesis_algorithm_comparison.json',
    ]
    expected_plots = [
        'plots/thesis_episode_wise_delay.svg',
        'plots/thesis_eta_vs_delay.svg',
        'plots/thesis_queue_vs_time.svg',
        'plots/thesis_algorithm_comparison.svg',
    ]
    for path in expected_raw + expected_plots:
        assert Path(path).exists()
    summary = Path('results/raw/thesis_algorithm_summary.csv').read_text()
    for label in ['Independent Learner - Full RL', 'Independent Learner - QPLF', 'Semi-Coordinated - Full RL', 'Semi-Coordinated - QPLF', 'Cyclic Queue Backpressure']:
        assert label in summary
    eta_text = Path('results/raw/thesis_eta_vs_delay.csv').read_text()
    assert ',0.1,' in eta_text and ',1.2,' in eta_text

def test_real_sumo_algorithm_comparison_helpers():
    from scripts.run_real_sumo_algorithm_comparison import ETAS, POLICY_LABELS, ALL_POLICIES, TRACI_START_RETRIES, softmax, integer_durations, choose_best_eta, parse_eta_values, simulation_metrics, sumo_command
    assert ETAS[0] == 0.1 and ETAS[-1] == 1.2 and len(ETAS) == 12
    for label in ['Independent Learner - Full RL', 'Independent Learner - QPLF', 'Semi-Coordinated - Full RL', 'Semi-Coordinated - QPLF', 'Cyclic Queue Backpressure']:
        assert label in POLICY_LABELS.values()
    probs = softmax([1.0, 2.0, 3.0], eta=0.5)
    assert len(probs) == 3 and abs(sum(probs) - 1.0) < 1e-9
    durations = integer_durations(probs, 80)
    assert sum(durations) == 80 and len(durations) == 3
    assert set(ALL_POLICIES) == set(POLICY_LABELS)
    assert TRACI_START_RETRIES == 3
    assert parse_eta_values("0.1, 0.5,1.2") == [0.1, 0.5, 1.2]
    metrics = simulation_metrics([10.0, 14.0], total_stops=4)
    assert metrics["completed_vehicles"] == 2
    assert metrics["mean_travel_time_seconds"] == 12.0
    assert metrics["average_stops_per_completed_vehicle"] == 2.0
    best = choose_best_eta(None, None, None, {"eta": 0.1, "mean_travel_time_seconds": None, "mean_network_queue": 8.0})
    assert best == (0.1, None, 8.0, "mean_network_queue")
    best = choose_best_eta(*best[:3], {"eta": 0.5, "mean_travel_time_seconds": 40.0, "mean_network_queue": 9.0})
    assert best == (0.5, 40.0, 8.0, "mean_travel_time_seconds")
    assert sumo_command("sumo", Path("scenario.sumocfg")) == ["sumo", "-c", "scenario.sumocfg", "--no-step-log", "true", "--no-warnings", "true"]
