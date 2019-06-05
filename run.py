# coding:utf-8
print('\n程序初始化中，请等待图像窗口弹出...\n')
print('注意!图像窗口弹出后请将鼠标移至图像窗口内，并点击一次!')
print('按w键速度加\n按s键速度减\n按空格键启动或停止小车')
print('按q键退出')

# 加载模块
import wiringpi
import cv2 as cv
import keras
import numpy as np 
import time 

# GPIO初始化
def gpioInit():
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(1, wiringpi.PWM_OUTPUT)  # 设置pwm输出
    wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
    wiringpi.pwmSetRange(12000)  # 设置频率
    wiringpi.pwmSetClock(32)
    wiringpi.pinMode(22, wiringpi.SOFT_PWM_OUTPUT) # 设置电机引脚软件pwm输出
    wiringpi.pinMode(23, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(24, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(25, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.softPwmCreate(22, 0, 100)  # 生成范围0-100的pwm
    wiringpi.softPwmCreate(23, 0, 100)
    wiringpi.softPwmCreate(24, 0, 100)
    wiringpi.softPwmCreate(25, 0, 100)


def setServoVal(val):
    if val < 380:
        val = 380
    if val > 620:
        val = 620
    wiringpi.pwmWrite(1, val)  # 设置舵机pwm值


# 设置电机pwm值
def setMotorVal(val):
    if val >= 0:
        wiringpi.softPwmWrite(22, val)
        wiringpi.softPwmWrite(23, 0)
        wiringpi.softPwmWrite(24, 0)
        wiringpi.softPwmWrite(25, val)
    else:
        wiringpi.softPwmWrite(22, 0)
        wiringpi.softPwmWrite(23, val)
        wiringpi.softPwmWrite(24, val)
        wiringpi.softPwmWrite(25, 0)


def main():
    start_flag = 0
    speed = 4
    gpioInit()
    capture = cv.VideoCapture(0)  # 启动摄像头
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 256)  # 设置拍摄图像大小
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 128)
    print('Loading model...')
    model = keras.models.load_model('model.h5')  # 加载训练好的模型
    print('Ready!')
    while True:
        # 按键键值获取和处理
        key = cv.waitKey(1) & 0xff  # 获取键值
        if key == ord('w'):
            speed = speed + 1  # 速度加
            if start_flag:
                setMotorVal(speed)  # 设置速度
        elif key == ord('s'):
            speed = speed - 1  #速度减
            if speed < 0:
                speed = 0
            if start_flag:
                setMotorVal(speed)  # 设置速度
        elif key == ord(' '):  # 启动或停止小车
            start_flag = ~start_flag 
            if start_flag:
                setMotorVal(speed)
            else:
                setMotorVal(0)
#        start = time.clock()
        ret, frame = capture.read()  # 获取一帧图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 转为灰度图
        img = cv.resize(gray, (240, 60))  # 转成240*60大小
        # 将图像转为网络可接受的输入大小
        img_data = []
        img = img.tolist()
        img_data.append(img)
        img_data = np.array(img_data, dtype=np.uint8)
        img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1).astype('float32')
        img_data = img_data / 255  # 归一化
        
        prediction = model.predict(img_data)  # 输入图像给模型，计算结果
        pred = int(240 * prediction[0] + 381)  # 转化为pwm值
#        end = time.clock()
#        print('time:' + str(end-start), '   pred:', str(pred))
        setServoVal(pred)  # 输出给舵机

        # 显示图像和转向角、速度、启停信息
        string = 'pre:' + str(pred) + ' speed:' + str(speed)
        if start_flag:
            string = string + ' run'
        else:
            string = string + ' stop'
        cv.putText(frame, string, (10,20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        # cv.namedWindow('img', 0)
        # cv.resizeWindow('img', 640, 480)
        cv.imshow("img", frame)
        if key == ord('q'):
            cv.destroyAllWindows() # 退出
            break


if __name__ == "__main__":
    main()
