from datetime import datetime, timedelta
from upbitlib.upbit import Upbit
import pandas as pd


def get_candles_data():
    access_key = 'YOUR_ACCESS_KEY'
    secret_key = 'YOUR_SECRET_KEY'

    # Upbit 객체 생성
    upbit = Upbit(access_key, secret_key)

    # 현재 시간
    now = datetime.now()

    # 시작 시간 설정 (2022년 1월 1일)
    start_time = datetime(2017, 10, 1)

    # 데이터를 저장할 파일 경로
    file_path = './daily_ETH_candles_2017.csv'

    # 데이터 가져오기
    with open(file_path, 'w') as file:
        # 헤더 작성
        # headers = "market,candle_date_time_utc,candle_date_time_kst,opening_price,high_price,low_price,trade_price,timestamp,candle_acc_trade_price,candle_acc_trade_volume,unit\n"
        headers = "market,candle_date_time_utc,candle_date_time_kst,opening_price,high_price,low_price,trade_price,timestamp,candle_acc_trade_price,candle_acc_trade_volume,prev_closing_price,change_price,change_rate\n"
        file.write(headers)

        while now > start_time:
            # to 파라미터 설정
            to = now.strftime('%Y-%m-%dT%H:%M:%S')


            # candles = upbit.get_candles_per_minutes(minute=30, market='KRW-BTC', to=to, count=200)
            candles = upbit.get_candles_daily("KRW-ETH", to=to, count=200)
            print(candles)

            # API 호출 결과 확인
            if candles is None:
                print(f"No data returned for {to}")
                continue

            # 가져온 데이터를 파일에 추가
            for candle in candles:
                data = ",".join([
                    candle['market'],
                    candle['candle_date_time_utc'],
                    candle['candle_date_time_kst'],
                    str(candle['opening_price']),
                    str(candle['high_price']),
                    str(candle['low_price']),
                    str(candle['trade_price']),
                    str(candle['timestamp']),
                    str(candle['candle_acc_trade_price']),
                    str(candle['candle_acc_trade_volume']),
                    str(candle['prev_closing_price']),
                    str(candle['change_price']),
                    str(candle['change_rate'])
                ])
                file.write(data + '\n')

            # 다음 조회를 위해 now 갱신
            # now -= timedelta(hours=200 * 30 / 60)
            now -= timedelta(days=201)

    print("Data saved to:", file_path)


# 실행
get_candles_data()
