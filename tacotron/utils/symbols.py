from . import cmudict

_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

_arpabet = ['@' + s for s in cmudict.valid_symbols]

symbols = [_pad, _eos] + list(_characters) + _arpabet
