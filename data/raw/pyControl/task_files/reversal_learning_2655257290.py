# Reversal learning task

from pyControl.utility import *
import hardware_definition as hw

# States and events.

states = [
    "initiation_state",
    "choice_state",
    "forced_choice_left",
    "forced_choice_right",
    "chose_left",
    "chose_right",
    "reward_left",
    "reward_right",
    "no_reward",
    "reward_consumption",
    "inter_trial_interval",
]

events = [
    "poke_4",
    "poke_4_out",
    "poke_5",
    "poke_5_out",
    "poke_6",
    "poke_6_out",
    "session_timer",
    "consumption_timer",
    "rsync",
]


initial_state = "initiation_state"

# Parameters

v.reward_vol_ul = 10  # Reward volume in ul.
v.hw_poke_4_calibration = {"slope": 7, "intercept": 22}  # Linear fit to solenoid calibration.
v.hw_poke_6_calibration = {"slope": 7, "intercept": 22}  # Linear fit to solenoid calibration.
v.good_prob = 0.8
v.bad_prob = 0.2
v.session_duration = 45 * minute  # Session duration.
v.ITI_duration = [1000, 3000]  # Inter trial interval duration (ms).
v.consumption_duration = 1 * second  # Time needed to stay out of reward port for consumption state to end.
v.post_choice_delay = 500 * ms  # Delay between choice and outcome.

v.threshold = 0.75  # Fraction correct choices to trigger reversal on next session.
v.tau = 8  # Time constant for moving average of choices (trials).
v.trials_post_threshold = [5, 15]  # Number of trials post threshold crossing till reversal [min, max]

v.click_volume = 10

# Variables

v.n_rewards = 0  # Number of rewards obtained.
v.n_trials = 0  # Number of trials recieved.
v.n_blocks = 0  # Number of blocks completed.
v.correct = None
v.good_poke = choice(["poke_4", "poke_6"])  # Which poke is currently good.
v.correct_mov_ave = exp_mov_ave(tau=v.tau, init_value=0.5)  # Exponential moving average of correct/incorrect choices.
v.forced_choice_sampler = sample_without_replacement([False, False, False, True])
v.forced_choice = False
v.threshold_crossed = False  # Set True when threshold for reversal crossed
v.trials_till_reversal = 0  # Set after threshold crossing to trigger reversal.

# State machine code --------------------------------------------------------------

# Run start and stop behaviour.


def run_start():
    # Set session timer
    set_timer("session_timer", v.session_duration)
    hw.speaker.set_volume(v.click_volume)


# State behaviour functions.


def initiation_state(event):
    # Turn on poke_5 LED and wait for initiation poke.
    if event == "entry":
        hw.poke_5.LED.on()
    elif event == "exit":
        hw.poke_5.LED.off()
    elif event == "poke_5":
        hw.speaker.click()
        v.forced_choice = v.forced_choice_sampler.next()
        if v.forced_choice:
            if withprob(0.5):
                goto_state("forced_choice_left")
            else:
                goto_state("forced_choice_right")
        else:
            goto_state("choice_state")


def choice_state(event):
    # Turn poke_4 and poke_6 LED on and wait for choice
    if event == "entry":
        hw.poke_4.LED.on()
        hw.poke_6.LED.on()
    elif event == "exit":
        hw.poke_4.LED.off()
        hw.poke_6.LED.off()
    elif event == "poke_4":
        hw.speaker.click()
        goto_state("chose_left")
    elif event == "poke_6":
        hw.speaker.click()
        v.choice = event
        goto_state("chose_right")


def forced_choice_left(event):
    # Turn poke_4 LED on and wait for choice
    if event == "entry":
        hw.poke_4.LED.on()
    elif event == "exit":
        hw.poke_4.LED.off()
    elif event == "poke_4":
        hw.speaker.click()
        goto_state("chose_left")


def forced_choice_right(event):
    # Turn poke_6 LED on and wait for choice
    if event == "entry":
        hw.poke_6.LED.on()
    elif event == "exit":
        hw.poke_6.LED.off()
    elif event == "poke_6":
        hw.speaker.click()
        goto_state("chose_right")


def chose_left(event):
    if event == "entry":
        v.choice = "poke_4"
        if get_trial_outcome():
            timed_goto_state("reward_left", v.post_choice_delay)
        else:
            timed_goto_state("no_reward", v.post_choice_delay)


def chose_right(event):
    if event == "entry":
        v.choice = "poke_6"
        if get_trial_outcome():
            timed_goto_state("reward_right", v.post_choice_delay)
        else:
            timed_goto_state("no_reward", v.post_choice_delay)


def reward_left(event):
    # Deliver reward to poke_4
    if event == "entry":
        timed_goto_state("reward_consumption", get_reward_duration(v.reward_vol_ul, v.hw_poke_4_calibration))
        hw.poke_4.SOL.on()
        hw.BNC_1.on()
    elif event == "exit":
        hw.poke_4.SOL.off()
        hw.BNC_1.off()


def reward_right(event):
    # Deliver reward to poke_6
    if event == "entry":
        timed_goto_state("reward_consumption", get_reward_duration(v.reward_vol_ul, v.hw_poke_6_calibration))
        hw.poke_6.SOL.on()
        hw.BNC_1.on()
    elif event == "exit":
        hw.poke_6.SOL.off()
        hw.BNC_1.off()


def reward_consumption(event):
    # Following reward, all LEDs are turned off, subject needs to stay out of poke_4 or 3 for consumption_duration before ITI starts.
    if event == "entry":
        if (not hw.poke_4.value()) and (not hw.poke_6.value()):  # Subject already exited port
            set_timer("consumption_timer", v.consumption_duration)
    elif event in ("poke_4_out", "poke_6_out"):
        set_timer("consumption_timer", v.consumption_duration)
    elif event in ("poke_4", "poke_6"):
        disarm_timer("consumption_timer")
    elif event == "consumption_timer":
        goto_state("inter_trial_interval")


def no_reward(event):
    # Deliver no reward
    if event == "entry":
        timed_goto_state("reward_consumption", 100 * ms)


def inter_trial_interval(event):
    # Go to init_state after randomly determined delay.
    if event == "entry":
        timed_goto_state("initiation_state", randint(*v.ITI_duration))


# State independent behaviour.


def all_states(event):
    # When 'session_timer' event occurs stop framework to end session.
    if event == "session_timer":
        stop_framework()


# Get trial outcome. -----------------------------------------------------------


def get_trial_outcome():
    # Function called after choice is made which determines trial outcome,
    # controls when reversals happen, and prints trial information.

    # Deterimine outcome and update moving average.
    v.correct = v.choice == v.good_poke
    v.outcome = withprob(v.good_prob if v.correct else v.bad_prob)
    if not v.forced_choice:
        v.correct_mov_ave.update(v.correct)

    # Determine when reversal occurs.
    if v.threshold_crossed:  # Subject has already crossed threshold.
        v.trials_till_reversal -= 1
        if v.trials_till_reversal == 0:  # Trigger reversal.
            v.good_poke = "poke_4" if (v.good_poke == "poke_6") else "poke_6"
            v.correct_mov_ave.value = 1 - v.correct_mov_ave.value
            v.threshold_crossed = False
            v.n_blocks += 1
    else:  # Check for threshold crossing.
        if v.correct_mov_ave.value > v.threshold:
            v.threshold_crossed = True
            v.trials_till_reversal = randint(*v.trials_post_threshold)

        # Print trial information.
    v.n_trials += 1
    v.n_rewards += v.outcome
    v.mov_ave = v.correct_mov_ave.value
    print_variables(
        [
            "n_trials",
            "n_rewards",
            "n_blocks",
            "forced_choice",
            "good_poke",
            "choice",
            "correct",
            "outcome",
            "mov_ave",
            "threshold_crossed",
        ]
    )

    return v.outcome


# Get reward duration. -----------------------------------------------------------


def get_reward_duration(reward_vol_ul, sol_calib):
    return reward_vol_ul * sol_calib["slope"] + sol_calib["intercept"]
