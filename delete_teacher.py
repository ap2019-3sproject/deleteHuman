from darkflow.net.build import TFNet
import cv2
import numpy as np
import copy


#分割したフレームが端であった場合端までの長さを返す
def judge_edge(length, unit, edge_n, n):
    if n == edge_n:
        return length
    else:
        return unit * (n + 1)


#フレームの分割結果を返す
def make_partial_frame(frame, split, height, width, frame_unit, i, j):
    return copy.deepcopy(frame[frame_unit[0] * i: judge_edge(height, frame_unit[0], split[0] - 1, i),
                               frame_unit[1] * j: judge_edge(width, frame_unit[1], split[1] - 1, j)])


#２つの分割が被っているかどうか
def judge_duplicate(s1, e1, s2, e2):
    if (s1 < s2 and e1 < s2) or (s1 > e2 and e1 > e2):
        return False
    return True


#人がいるかどうか
def judge_person(result, split, height, width, frame_unit, i, j):
    #yoloの検知する長方形が人より小さいため、バッファを取る
    buffer = 50
    for item in result:
        tlx = item['topleft']['x'] - buffer
        tly = item['topleft']['y'] - buffer
        brx = item['bottomright']['x'] + buffer
        bry = item['bottomright']['y'] + buffer
        label = item['label']
        conf = item['confidence']
        if conf > 0.6 and label == 'person':
            if judge_duplicate(tly, bry, frame_unit[0] * i, judge_edge(height, frame_unit[0], split[0] - 1, i)) and judge_duplicate(tlx, brx, frame_unit[1] * j, judge_edge(width, frame_unit[1], split[1] - 1, j)):
                return True
    return False


def main():
    options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.1}
    tfnet = TFNet(options)

    #分割数の指定
    split = [4, 20]

    #動画ファイルの読み込み
    cap = cv2.VideoCapture('sample.mp4')

    #開始時間の指定
    shift_time = 0
    cap.set(0,shift_time*1000)

    ret, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]
    print(f'height: {height}')
    print(f'width: {width}')

    back = cv2.imread('back.png')
    back_height = back.shape[0]
    back_width = back.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #出力ファイルの指定
    out = cv2.VideoWriter('sample_output.mp4', fourcc, 30.0, (back_width, back_height))
    frame_unit = [height // split[0], width // split[1]]
    frame_list = [[make_partial_frame(frame, split, height, width, frame_unit, i, j) for j in range(split[1])] for i in range(split[0])]

    count = 1
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        result = tfnet.return_predict(frame)
        for i in range(split[0]):
            for j in range(split[1]):
                if not judge_person(result, split, height, width, frame_unit, i, j):
                    frame_list[i][j] = make_partial_frame(frame, split, height, width, frame_unit, i, j)
                frame[frame_unit[0] * i: judge_edge(height, frame_unit[0], split[0] - 1, i), frame_unit[1] * j: judge_edge(width, frame_unit[1], split[1] - 1, j)] = frame_list[i][j]

        cv2.imshow("Show FLAME Image", frame)
        output_frame = copy.deepcopy(back)
        output_frame[0: height, 0: width] = frame
        character = cv2.imread(f'photoes/{count}.png')
        ch_height = character.shape[0]
        ch_width = character.shape[1]
        output_frame[back_height - ch_height: back_height, back_width - ch_width: back_width] = character
        out.write(output_frame)
        #'q'で終了
        k = cv2.waitKey(10);
        if k == ord('q'):  break;
        count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
