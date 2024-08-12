from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def dl_signal(input_data):
    model = load_model('../model/model_ws2_lr0.01_bs128_m1.h5')
    predictions = model.predict(input_data)*1.016
    # print(predictions[10:])
    plt.plot(predictions[-5:])
    plt.show()

    short_ma = np.mean(predictions[-5:])
    long_ma = np.mean(predictions[-10:])

    prev_short_ma = np.mean(predictions[-6:-1])
    prev_long_ma = np.mean(predictions[-11:-6])

    print('short ma: ', short_ma)
    print('long ma: ', long_ma)

    if prev_short_ma < prev_long_ma and short_ma > long_ma:  # cross over
        print("매수 시그널 발생! (cross over)")
        return 1
    elif prev_short_ma > prev_long_ma and short_ma < long_ma:  # cross below
        print("매도 시그널 발생! (cross below)")
        return -1
    else:
        return 0


def rsi_signal(data):
    rsi = data['rsi']
    last_rsi = rsi.iloc[-1]
    print("RSI: ", last_rsi)
    if last_rsi > 78:
        print("매도 시그널 발생! (RSI 상승)")
        return -1
    elif last_rsi < 30:
        print("매수 시그널 발생! (RSI 하락)")
        return 1
    else:
        return 0
