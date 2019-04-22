from c_spsp_base import Penguin


if __name__ == '__main__':
    normal_penguin = Penguin('fish')
    fast_penguin = Penguin.__new__(Penguin, 'wheat')
    # note: not calling __init__() !