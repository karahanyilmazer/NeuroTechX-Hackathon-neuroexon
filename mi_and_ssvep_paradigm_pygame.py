# %%
"""
Motor imagery paradigm based on a combination of the Graz paradigm and the
Berlin paradigm implemented in Pygame. 

Author:
    Karahan Yilmazer

Email:
    yilmazerkarahan@gmail.com
"""

import numpy as np
from screeninfo import get_monitors
import pygame as pg
from threading import Barrier, Thread
from math import sin, pi
from pylsl import local_clock, StreamInfo, StreamOutlet

# %%
# Define custom colors
GREY = (103, 103, 110)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

monitors = get_monitors()

if len(monitors) == 2:
    idx = 1
    RATIO = 1.1
else:
    idx = 0
    RATIO = 1.7

# Get the screen size of the only monitor
SCREEN_WIDTH = monitors[idx].width
SCREEN_HEIGHT = monitors[idx].height

# Calculate the Pygame window size
WINDOW_WIDTH = SCREEN_WIDTH / RATIO
WINDOW_HEIGHT = SCREEN_HEIGHT / RATIO

# Get the center of the screen
CENTER_X = WINDOW_WIDTH / 2
CENTER_Y = WINDOW_HEIGHT / 2

# Parameters for the fixation cross
FIX_LENGTH = WINDOW_HEIGHT / 2
FIX_ARM_LENGTH = WINDOW_HEIGHT / 4
FIX_ARM_WIDTH = 12

# Parameters for the arrow shown as cue
ARROW_LENGTH = 100
ARROW_WIDTH = 80
CUE_ARM_WIDTH = 60
CUE_ARM_LENGTH = FIX_ARM_LENGTH - ARROW_LENGTH + 1


class Paradigm(object):

    def __init__(self,
                 window,
                 durations,
                 cues,
                 freq_mapping,
                 bg_color=GREY,
                 stim_color=WHITE,
                 cue_color=RED):

        self.window = window
        self.durations = durations
        self.cues = cues
        self.freq_mapping = freq_mapping
        self.bg_color = bg_color
        self.stim_color = stim_color
        self.cue_color = cue_color
        self.tot_trials = None
        self.tot_blocks = None
        self.thread_running = True
        self.window_open = True

        # Define the marker stream
        info = StreamInfo(name='pygame_markers',
                          type='Markers',
                          channel_count=1,
                          nominal_srate=0,
                          channel_format='string',
                          source_id='pygame_markers')
        self.marker_stream = StreamOutlet(info)

        # Initialize the multithreading setup
        self.barrier = Barrier(len(self.freq_mapping))
        self.threads = []

        # Surface for the cross to be shown in
        self.fix_surf = pg.Surface((FIX_LENGTH, FIX_LENGTH))
        self.fix_surf_rect = self.fix_surf.get_rect(center=(CENTER_X, CENTER_Y))

        # Fixation vertices
        self.fix_left = pg.Rect(0, 0, FIX_ARM_LENGTH, FIX_ARM_WIDTH)
        self.fix_left.midright = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)

        self.fix_right = pg.Rect(0, 0, FIX_ARM_LENGTH, FIX_ARM_WIDTH)
        self.fix_right.midleft = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)

        self.fix_top = pg.Rect(0, 0, FIX_ARM_WIDTH, FIX_ARM_LENGTH)
        self.fix_top.midbottom = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)

        self.fix_bottom = pg.Rect(0, 0, FIX_ARM_WIDTH, FIX_ARM_LENGTH)
        self.fix_bottom.midtop = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)

        # Cue vertices
        self.cue_left = pg.Rect(0, 0, CUE_ARM_LENGTH, CUE_ARM_WIDTH)
        self.cue_left.midright = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)
        self.left_arrow_vertices = ((0, FIX_ARM_LENGTH),
                                    (ARROW_LENGTH,
                                     FIX_ARM_LENGTH - ARROW_WIDTH),
                                    (ARROW_LENGTH,
                                     FIX_ARM_LENGTH + ARROW_WIDTH))

        self.cue_right = pg.Rect(0, 0, CUE_ARM_LENGTH, CUE_ARM_WIDTH)
        self.cue_right.midleft = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)
        self.right_arrow_vertices = ((FIX_LENGTH, FIX_ARM_LENGTH),
                                     (FIX_LENGTH - ARROW_LENGTH,
                                      FIX_ARM_LENGTH - ARROW_WIDTH),
                                     (FIX_LENGTH - ARROW_LENGTH,
                                      FIX_ARM_LENGTH + ARROW_WIDTH))

        self.cue_bottom = pg.Rect(0, 0, CUE_ARM_WIDTH, CUE_ARM_LENGTH)
        self.cue_bottom.midtop = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)
        self.bottom_arrow_vertices = ((FIX_ARM_LENGTH, FIX_LENGTH),
                                      (FIX_ARM_LENGTH - ARROW_WIDTH,
                                       FIX_LENGTH - ARROW_LENGTH),
                                      (FIX_ARM_LENGTH + ARROW_WIDTH,
                                       FIX_LENGTH - ARROW_LENGTH))

        self.cue_top = pg.Rect(0, 0, CUE_ARM_WIDTH, CUE_ARM_LENGTH)
        self.cue_top.midbottom = (FIX_ARM_LENGTH, FIX_ARM_LENGTH)
        self.top_arrow_vertices = ((FIX_ARM_LENGTH,
                                    0), (FIX_ARM_LENGTH - ARROW_WIDTH,
                                         0 + ARROW_LENGTH),
                                   (FIX_ARM_LENGTH + ARROW_WIDTH,
                                    0 + ARROW_LENGTH))

    def clear_screen(self):
        # Reset the fixation surface
        self.fix_surf.fill(GREY)
        # Clear the screen
        self.window.fill(GREY)

        # Update the screen
        pg.display.flip()

    def show_fix(self):
        # Draw the fixation cross
        pg.draw.rect(self.fix_surf, WHITE, self.fix_top)
        pg.draw.rect(self.fix_surf, WHITE, self.fix_bottom)
        pg.draw.rect(self.fix_surf, WHITE, self.fix_left)
        pg.draw.rect(self.fix_surf, WHITE, self.fix_right)

        self.window.blit(self.fix_surf, self.fix_surf_rect)

        # Update the screen
        pg.display.flip()

    def show_cue(self, cue):
        if cue == 'left':

            pg.draw.rect(self.fix_surf, GREY, self.fix_left)
            pg.draw.rect(self.fix_surf, WHITE, self.cue_left)
            pg.draw.polygon(self.fix_surf, WHITE, self.left_arrow_vertices)

        if cue == 'right':

            pg.draw.rect(self.fix_surf, GREY, self.fix_right)
            pg.draw.rect(self.fix_surf, WHITE, self.cue_right)
            pg.draw.polygon(self.fix_surf, WHITE, self.right_arrow_vertices)

        if cue == 'feet':

            pg.draw.rect(self.fix_surf, GREY, self.fix_bottom)
            pg.draw.rect(self.fix_surf, WHITE, self.cue_bottom)
            pg.draw.polygon(self.fix_surf, WHITE, self.bottom_arrow_vertices)

        if cue == 'tongue':

            pg.draw.rect(self.fix_surf, GREY, self.fix_top)
            pg.draw.rect(self.fix_surf, WHITE, self.cue_top)
            pg.draw.polygon(self.fix_surf, WHITE, self.top_arrow_vertices)

        if cue == 'rest':
            self.show_fix()

        self.window.blit(self.fix_surf, self.fix_surf_rect)

        pg.display.flip()

    def get_beep(self, size_format, f=440, duration=0.5):
        # Reference: https://github.com/psychopy/psychopy/blob/release/psychopy/sound/_base.py

        # Sampling frequency in Hz
        fs = 44100
        # Number of samples in the sine wave
        n_samples = int(duration * fs)
        # Calculate the length of the Hanning window
        han_win_size = int(min(fs // 200, n_samples // 15))
        # Get the Hanning window
        han_win = np.hanning(2 * han_win_size + 1)

        # Create the sine wave
        sine_wave = np.arange(0, 1, 1 / n_samples)
        sine_wave *= 2 * np.pi * f * duration
        sine_wave = np.sin(sine_wave)

        # Apply tapering for more pleasant on- and offset
        sine_wave[:han_win_size] *= han_win[:han_win_size]
        sine_wave[-han_win_size:] *= han_win[han_win_size + 1:]

        # Convert the type to match Pygame settings
        if size_format == -16:
            sine_wave = (sine_wave * 2**15).astype(np.int16)
        elif size_format == 16:
            sine_wave = ((sine_wave + 1) * 2**15).astype(np.uint16)
        elif size_format == -8:
            sine_wave = (sine_wave * 2**7).astype(np.Int8)
        elif size_format == 8:
            sine_wave = ((sine_wave + 1) * 2**7).astype(np.uint8)

        return sine_wave

    def get_cues(self):
        tot_trials = self.tot_trials
        cues = self.cues

        # Create an empty list for the order of cues to run through
        pick_list = []

        if tot_trials is not None:
            if tot_trials < len(cues):
                raise ValueError(
                    'Number of trials should be at least the same as the number of cues!'
                )

        # Get the number of cues
        len_cues = len(cues)
        # Get the quotient of the division of desired number of trials and the
        # number of cues
        tmp = tot_trials // len_cues
        # Multiply the cues list with the quotient to get equal number of instances for each cue
        pick_list = cues * tmp
        # Get the remainder of the division
        remainder = tot_trials % len_cues
        # If the remainder is not 0
        if remainder != 0:
            # Add random cues to the list until the desired number of trials is reached
            for _ in range(remainder):
                pick_list.append(np.random.choice(cues))

        # Shuffle the cue order for randomness
        np.random.shuffle(pick_list)

        return pick_list

    def run_exp(self, tot_trials, tot_blocks):
        # Set the class attributes
        self.tot_trials = tot_trials
        self.tot_blocks = tot_blocks

        # Initialize the number of trials and blocks
        trial_num = 1
        block_num = 1

        # Get the durations
        durations = self.durations

        # Get the window
        window = self.window

        # Initialize the flags for different stages of the experiment
        welcome = True
        trial_start_1 = False
        trial_start_2 = False
        show_cue = False
        trial_break = False
        block_break = False

        window_open = True
        run_exp = False
        exp_end = False

        # Create a font for displaying text
        msg_font = pg.font.Font(pg.font.get_default_font(), 50)

        # Define the messages to be shown
        msg_exp_start = msg_font.render(
            'Welcome to the recording. Press any key to continue.', False,
            WHITE)
        msg_exp_end = msg_font.render(
            'Press key UP to start a new block, key DOWN to end the recording.',
            False, WHITE)

        # Get the rectangle surrounding the text
        msg_exp_start_rect = msg_exp_start.get_rect()
        msg_exp_end_rect = msg_exp_end.get_rect()

        # Set the center of the rectangular object
        msg_exp_start_rect.center = (CENTER_X, CENTER_Y)
        msg_exp_end_rect.center = (CENTER_X, CENTER_Y)

        # Display the welcome message
        self.window.blit(msg_exp_start, msg_exp_start_rect)
        # Update the screen
        pg.display.flip()

        # Initialize the mixer to play sounds
        pg.mixer.init()
        # Get mixer settings to pass to get_beep()
        _, size_format, _ = pg.mixer.get_init()
        # Get the beep
        beep = pg.mixer.Sound(
            self.get_beep(size_format, f=440, duration=durations['beep']))

        while window_open:

            # Get all the current events
            for event in pg.event.get():
                # If the window is closed
                if event.type == pg.QUIT:
                    pg.display.quit()  # Kill the window
                    window_open = False  # Exit the while loop

                # If a key is pressed
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.display.quit()  # Kill the window
                        window_open = False  # Exit the while loop
                    else:
                        if welcome:
                            # Empty the screen
                            self.clear_screen()
                            pg.display.flip()

                            # Get a list of cues to be shown
                            cues = self.get_cues()
                            print(f'Block {block_num}: {cues}')

                            # Get out of the welcoming stage
                            welcome = False
                            # Start window_open the experiment
                            run_exp = True
                            trial_start_1 = True

                        # If the total amount of blocks has been reached
                        elif exp_end:
                            if event.key == pg.K_DOWN:
                                pg.display.quit()  # Kill the window
                                window_open = False  # Exit the while loop

                            elif event.key == pg.K_UP:
                                # Clear the window
                                self.clear_screen()
                                # Get a new set of cues
                                cues = self.get_cues()
                                print(f'Block {block_num}: {cues}')
                                
                                # Reset the flags
                                run_exp = True
                                exp_end = False
                                trial_start_1 = True

            if run_exp:

                # TRIAL START
                # ==============================================================
                if trial_start_1:
                    # Show the fixation cross
                    self.show_fix()

                    # Push the marker to the LSL stream
                    self.marker_stream.push_sample(
                        [f'trial_begin_{trial_num}-{local_clock()}'])
                    # print([f'trial_begin_{trial_num}-{local_clock()}'])

                    # Start the timer
                    timer_start = local_clock()
                    # Get the waiting duration
                    wait_time = durations['trial_begin_1']

                    # Set the flags
                    run_exp = False
                    trial_start_1 = False
                    trial_start_2 = True

                elif trial_start_2:
                    # Play the sound
                    beep.play()

                    # Push the marker to the LSL stream
                    self.marker_stream.push_sample([f'beep-{local_clock()}'])
                    # print([f'beep-{local_clock()}'])

                    # Start the timer
                    timer_start = local_clock()
                    # Get the waiting duration
                    wait_time = durations['trial_begin_2']

                    # Set the flags
                    run_exp = False
                    trial_start_2 = False
                    show_cue = True

                # CUE
                # ==============================================================
                elif show_cue:
                    # Get the current cue
                    cue = cues[trial_num - 1]
                    # Display the cue
                    self.show_cue(cue)

                    # Push the marker to the LSL stream
                    self.marker_stream.push_sample(
                        [f'cue_{cue}-{local_clock()}'])
                    # print([f'cue_{cue}-{local_clock()}'])

                    # Start the timer
                    timer_start = local_clock()
                    # Get the waiting duration
                    wait_time = durations['cue']

                    # If the total number of trials within a block is reached
                    if trial_num == tot_trials:
                        # If there are still blocks to run
                        if block_num < tot_blocks:
                            # Set the trial number to 0 as it will be incremented at the
                            # end of the loop
                            trial_num = 0
                            # Set the flag to show the block end message
                            block_break = True
                        # If there are no more blocks to run
                        else:
                            # Reset the trial number in case another block is chosen to be
                            # run
                            trial_num = 0
                            # Set the flag to show the block end message
                            block_break = True
                    # If there are still trials to run in the current block
                    else:
                        # Set the flag to go to the block break
                        trial_break = True

                    # Increment the trial number
                    trial_num += 1

                    # Set the flag
                    run_exp = False
                    show_cue = False

                # TRIAL BREAK
                # ==============================================================
                elif trial_break:
                    # Clear the screen
                    self.clear_screen()

                    # Push the marker to the LSL stream
                    self.marker_stream.push_sample([f'pause-{local_clock()}'])
                    # print([f'pause-{local_clock()}'])

                    # Start the timer
                    timer_start = local_clock()
                    # Get the random pause duration
                    wait_time = durations['trial_end'] + np.round(
                        np.random.uniform(0, 1), 2)

                    # Set the flags
                    run_exp = False
                    trial_break = False
                    trial_start_1 = True

                # BLOCK BREAK
                # ==============================================================
                elif block_break:

                    # Prepare the message
                    msg_block_end = msg_font.render(
                        f'Block #{block_num} has ended.', False, WHITE)
                    msg_block_end_rect = msg_block_end.get_rect()
                    msg_block_end_rect.center = (CENTER_X, CENTER_Y)

                    # Empty the screen
                    self.clear_screen()
                    # Show the message
                    window.blit(msg_block_end, msg_block_end_rect)
                    # Update the screen
                    pg.display.flip()

                    # Push the marker to the LSL stream
                    self.marker_stream.push_sample(
                        [f'block_end_{block_num}-{local_clock()}'])
                    # print([f'block_end_{block_num}-{local_clock()}'])

                    # Start the timer
                    timer_start = local_clock()
                    # Get the waiting duration
                    wait_time = durations['block_end']

                    # Increment the block number
                    block_num += 1
                    # Set the flags
                    block_break = True
                    run_exp = False

                    # If the total number of blocks is exceeded
                    if block_num > tot_blocks:
                        self.clear_screen()
                        # Show the message
                        window.blit(msg_exp_end, msg_exp_end_rect)
                        # Update the screen
                        pg.display.flip()
                        # Set the flags
                        exp_end = True
                        run_exp = False
                        block_break = False
                    else:
                        # Get a new set of cues
                        cues = self.get_cues()
                        print(f'Block {block_num}: {cues}')
                        # Set the flag
                        trial_start_1 = True

            else:
                if not welcome:
                    if local_clock() - timer_start < wait_time:
                        pass
                    else:
                        run_exp = True


# %%
# Define the durations for each part of the paradigm
paradigm_durations = {
    'trial_begin_1': 1,
    'trial_begin_2': 1,
    'beep': 0.5,
    'cue': 4,
    'trial_end': 2,
    'block_end': 4
}

# Define the cues that are going to be shown
# mi_cues = ['rest', 'left', 'right', 'feet', 'tongue']
# mi_cues = ['rest', 'left', 'right']
# mi_cues = ['rest', 'right']
# mi_cues = ['rest']
mi_cues = ['right']

# Define the frequency mapping ofs the stimulation boxes
freq_mapping = {'rest': 0, 'left': 9, 'right': 15}

# Initialize Pygame
pg.init()

# Create a Pygame window
window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
pg.display.set_caption('Motor Imagery + SSVEP Paradigm')
window.fill(GREY)

# Initialize the Paradigm object to manage the functions and variables
paradigm = Paradigm(window, paradigm_durations, mi_cues, freq_mapping)

# Run the experiment
paradigm.run_exp(tot_trials=30, tot_blocks=2)
# %%
