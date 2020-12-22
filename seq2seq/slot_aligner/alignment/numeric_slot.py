import re

from seq2seq.slot_aligner.alignment.utils import find_first_in_list


def align_numeric_slot(text, text_tok, slot, value):
    number_to_word = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
        '10': 'ten',
        '11': 'eleven',
        '12': 'twelve',
        '13': 'thirteen',
        '14': 'fourteen',
        '15': 'fifteen',
        '16': 'sixteen',
        '17': 'seventeen',
        '18': 'eighteen',
        '19': 'nineteen',
        '20': 'twenty',
        '30': 'thirty',
        '40': 'forty',
        '50': 'fifty',
        '60': 'sixty',
        '70': 'seventy',
        '80': 'eighty',
        '90': 'ninety',
        '100': 'hundred',
    }
    word_to_number = {val: key for key, val in number_to_word.items()}

    match = re.search(fr'\b{value}\b', text)
    if match:
        return match.start()

    for value_word in value.split():
        for number_map in [number_to_word, word_to_number]:
            value_alt = number_map.get(value_word)
            if value_alt:
                match = re.search(fr'\b{value_alt}\b', text)
                if match:
                    return match.start()

    return -1


def align_numeric_slot_with_unit(text, text_tok, slot, value):
    value_number = value.split(' ')[0]
    try:
        float(value_number)
    except ValueError:
        return -1

    _, pos = find_first_in_list(value_number, text_tok)

    return pos


def align_year_slot(text, text_tok, slot, value):
    try:
        int(value)
    except ValueError:
        return -1

    year_alternatives = [value]
    if len(value) == 4:
        year_alternatives.append('\'' + value[-2:])
        year_alternatives.append(value[-2:])

    for val in year_alternatives:
        if len(val) > 2:
            pos = text.find(val)
        else:
            _, pos = find_first_in_list(val, text_tok)

        if pos >= 0:
            return pos

    return -1
