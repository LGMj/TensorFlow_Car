import wiringpi
import pygame
import cv2 as cv
import keras
import numpy as np 
import time 

def gpioInit():
    wiringpi.wiringPiSetup()
    wiringpi.pinMode(1, wiringpi.PWM_OUTPUT)
    wiringpi.pwmSetMode(wiringpi.PWM_MODE_MS)
    wiringpi.pwmSetRange(12000)
    wiringpi.pwmSetClock(32)
    wiringpi.pinMode(22, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(23, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(24, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.pinMode(25, wiringpi.SOFT_PWM_OUTPUT)
    wiringpi.softPwmCreate(22, 0, 100)
    wiringpi.softPwmCreate(23, 0, 100)
    wiringpi.softPwmCreate(24, 0, 100)
    wiringpi.softPwmCreate(25, 0, 100)


def setServoVal(val):
    if val < 380:
        val = 380
    if val > 620:
        val = 620
    wiringpi.pwmWrite(1, val)


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
    pygame.init()
    gpioInit()
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 256)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 128)
    screen = pygame.display.set_mode((0, 0), 0, 32)
    screen_rect = screen.get_rect()
    model = keras.models.load_model('model.h5')
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("exit")
                    cv.destroyAllWindows()
                    exit()
                elif event.key == pygame.K_w:
                    speed = speed + 1
                elif event.key == pygame.K_s:
                    speed = speed - 1
                else:
                    pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                start_flag = ~start_flag

            if start_flag:
                setMotorVal(speed)
            else:
                setMotorVal(0)
            print(speed)
        
        start = time.clock()
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.resize(gray, (240, 60))
        img_data = []
        img = img.tolist()
        img_data.append(img)
        img_data = np.array(img_data, dtype=np.uint8)
        img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1).astype('float32')
        img_data = img_data / 255
        
        prediction = model.predict(img_data)
        pred = int(240 * prediction[0] + 381)
        end = time.clock()
        print('time:' + str(end-start))
        setServoVal(pred)

        string = 'pre:' + str(pred)
        cv.putText(frame, string, (10,20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.imshow("img", frame)



if __name__ == "__main__":
    main()
