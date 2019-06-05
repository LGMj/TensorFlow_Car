# coding:utf-8
# 加载模块
import wiringpi
import pygame
import cv2 as cv

# 初始化GPIO
def gpioInit():
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(1, wiringpi.PWM_OUTPUT)  # 设置舵机pwm输出
    wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
    wiringpi.pwmSetRange(12000)  # 设置频率
    wiringpi.pwmSetClock(32)
    wiringpi.pinMode(22, wiringpi.SOFT_PWM_OUTPUT)  # 设置电机软件pwm输出
    wiringpi.pinMode(23, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(24, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(25, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.softPwmCreate(22, 0, 100)  # 输出0-100的pwm
    wiringpi.softPwmCreate(23, 0, 100)
    wiringpi.softPwmCreate(24, 0, 100)
    wiringpi.softPwmCreate(25, 0, 100)


# 设置舵机pwm值
def setServoVal(val):
    if val < 380:
        val = 380
    if val > 620:
        val = 620
    wiringpi.pwmWrite(1, val)


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
    index = 0
    pygame.init()  # 初始化pygame模块
    gpioInit()  # 初始化GPIO
    capture = cv.VideoCapture(0)  #打开摄像头
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 256)  # 设置图像大小
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 128)
    screen = pygame.display.set_mode((0, 0), 0, 32)  # 创建pygame窗口
    screen_rect = screen.get_rect()  # 获取窗口大小 
    while True:
        # 按键处理 
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("exit")  # ESC键退出
                    exit()
                elif event.key == pygame.K_w:
                    speed = speed + 1  # w键加速
                elif event.key == pygame.K_s:
                    speed = speed - 1  # s键减速
                else:
                    pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                start_flag = ~start_flag  # 点击屏幕启动或停止小车

            if start_flag:
                setMotorVal(speed)  # 设置电机pwm值
            else:
                setMotorVal(0)
            print(speed)
        x, y = pygame.mouse.get_pos() # 获取鼠标位置
        servo_val = 620 - int(x*240 / screen_rect.width)  # 将鼠标X轴的位置转为舵机pwm值 
        setServoVal(servo_val)  # 设置舵机pwm值
        ret, frame = capture.read()  # 读取一帧图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 转为灰度图
        if start_flag:
            cv.imwrite('./image/' + str(index) + '_' + str(x) + '.jpeg', gray)  # 保存图像和转角值
        index = index + 1
        wiringpi.delay(50)  # 延时50ms


if __name__ == "__main__":
    main()
