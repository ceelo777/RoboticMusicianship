import itertools
from NoteStateBatches import *
def getOrDefault(state, index, default):
    try:
        return state[index]
    except IndexError:
        return default

def build_context(state):
    context = []
    for i in range(12):
        context.append(0)
    for note, notestate in enumerate(state):
        if (notestate[0] == 1):
            pitch_class = (note + lower_bound) % 12
            context[pitch_class] += 1
    return context

def build_beat(time):
    times = [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]
    beat = []
    for t in times:
        beat.append(2 * t - 1)
    return beat

def note_input_form(note, state, context, beat):
    position = note
    pos_arr = []
    pos_arr.append(position)

    # The pitch class is the set of all pitches that are a whole number of octaves apart
    pitch_class = (note + lower_bound) % 12

    part_pitch = []    
    for i in range(12):
        if (i == pitch_class):
            part_pitch.append(1)
        else:
            part_pitch.append(0)

    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable((getOrDefault(state, note + i, [0, 0]) for i in range(-12, 13))))

    # Create the part context for the note
    part_context = context[pitch_class:] + context[:pitch_class]
    value = pos_arr + part_pitch + part_prev_vicinity + part_context + beat + [0]
    # print("Position Array: ", pos_arr)
    # print("Part Pitch: ", part_pitch)
    # print("Part Previous Vicinity: ", part_prev_vicinity)
    # print("Part Context: ", part_context)
    # print("Beat: ", beat)
    # print("Value: ", value)
    return value
    

def nsb_note_to_input(time, state):
    beat = build_beat(time)
    context = build_context(state)
    input_note = []
    for note in range(len(state)):
        input_note.append(note_input_form(note, state, context, beat))        
    return input_note

def nsb_to_input(nsb_segment):
    piece_segment = []
    for time, state in enumerate(nsb_segment):
        # print("Time: ", time)
        # print("State: ", state)
        piece_segment.append(nsb_note_to_input(time, state))
    return piece_segment
