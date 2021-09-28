from pynput import keyboard
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import time
import datetime
import threading
import pandas as pd
import joblib
import matplotlib.pyplot as plt


"""Done By Mohammed Al Haimi"""

# visual the data into digrams and entery the data to the machine learning


class DataTrain():
    def __init__(self, data_file):
        self.pass_data = pd.read_csv(data_file)
        self.X = self.pass_data.drop(columns=['user'])
        self.Y = self.pass_data['user']
        self.model = DecisionTreeClassifier()
        print("Reading csv .... ")
        print("putting data into Machine learning Model ... ")
        self.train()

    def main(self):

        print("_"*20)
        print("1 Test Model by your self ")
        print("2 show the accuracy Score ")
        print("3 go back the menu")
        print("4 Draw digram")
        print("0 exit ")
        choice = input("Choose a number ^ : ")

        if (choice == 1 or choice == '1'):
            print("Starting manual learning mode")
            self.test_manual()
        elif (choice == 2 or choice == '2'):
            # testing_mode()
            self.accuracy_rate()
        elif (choice == 3 or choice == "3"):
            main()
        elif (choice == 4 or choice == "4"):
            self.visualizer()
        elif(choice == 0 or choice == '0'):
            exit()
        else:
            main()

    def visualizer(self):

        plt.scatter(self.X['release-0'], self.Y, )
        plt.xlabel('Users')
        plt.ylabel('timestamp in miliseconds')
        plt.title('Scatter plot on typing pattern')
        plt.show()

    def test_manual(self):

        p = PasswordLearning('0')
        time.sleep(1)
        with keyboard.Listener(
                on_press=p.on_press,
                on_release=p.on_release) as Listen:
            test = p.testing()
        predict = self.model.predict([test])
        print(predict)
        self.main()

    def accuracy_rate(self):

        x_train, x_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.2)
        predict = self.model.predict(x_test)
        score = accuracy_score(y_test, predict)
        print(f'accuracy socre is {score}')
        self.main()

    def train(self):

        self.model.fit(self.X, self.Y)
        joblib.dump(self.model, 'password-patterns.joblib')
        tree.export_graphviz(self.model, out_file='password-strock.dot',
                             feature_names=['press-0', 'release-0', 'press-1', 'release-1', 'press-2', 'release-2', 'press-3', 'release-3', 'press-4', 'release-4', 'press-5',
                                            'release-5', 'press-6', 'release-6', 'press-7', 'release-7', 'press-8', 'release-8', 'press-9', 'release-9', 'press-10', 'release-10', 'press-11', 'release-11', 'press-12', 'release-12'],

                             label='all',
                             rounded=True,
                             filled=True)


# class to run the listner functions and store the typing data

class PasswordLearning():

    def __init__(self, userName):

        self.PRESSED_TIME = None
        self.counter_key = 0
        self.time_pressed = []
        self.time_released = []
        self.time_btween = []
        self.total_time = None
        self.START_TIME = None
        self.RELEASED_TIME = None
        self.user_id = userName
        self.pwd = None
        self.csvObjects = {}
        self.objectList = []
        self.test = []

    def on_press(self, key):

        if self.counter_key == 0:

            self.START_TIME = self.TimestampMillisec64()
            self.PRESSED_TIME = self.TimestampMillisec64()
            self.counter_key += 1
            self.time_pressed.append(0)
        elif self.counter_key < 12:

            self.PRESSED_TIME = self.TimestampMillisec64()
            self.counter_key += 1
            self.time_pressed.append((self.PRESSED_TIME - self.START_TIME))
        elif self.counter_key == 12:

            self.PRESSED_TIME = self.TimestampMillisec64()
            self.time_pressed.append((self.PRESSED_TIME - self.START_TIME))
            self.counter_key += 1
        elif self.counter_key <= 13:

            return False

    def on_release(self, key):

        if self.counter_key < 13:

            end = self.TimestampMillisec64()
            self.RELEASED_TIME = (end - self.PRESSED_TIME)
            self.time_released.append(self.RELEASED_TIME)
        elif self.counter_key == 13:

            end = self.TimestampMillisec64()
            self.RELEASED_TIME = (end - self.PRESSED_TIME)
            self.time_released.append(self.RELEASED_TIME)
            total = self.TimestampMillisec64()
            self.total_time = (total - self.START_TIME)
        else:

            return False

    def TimestampMillisec64(self):

        return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

    def time_btween_keys(self):

        for index, timePressed in enumerate(self.time_pressed):

            if timePressed == self.time_pressed[0]:

                self.csvObjects['user'] = self.user_id
            if timePressed == self.time_pressed[-1]:

                self.time_released[index] = self.time_released[index] + timePressed
                keypress = "press-" + str(index)
                self.csvObjects[keypress] = timePressed
                keyreleased = "release-" + str(index)
                self.csvObjects[keyreleased] = self.time_released[index]
                break
            else:

                btwn = self.time_pressed[index + 1] - \
                    (timePressed + self.time_released[index])
                self.time_btween.append(btwn)
                self.time_released[index] = self.time_released[index] + timePressed
            keypress = "press-" + str(index)
            self.csvObjects[keypress] = timePressed
            keyreleased = "release-" + str(index)
            self.csvObjects[keyreleased] = self.time_released[index]

        self.test.append([valu for attr, valu in self.csvObjects.items()])
        self.time_pressed = []
        self.time_released = []
        self.time_btween = []

    def password(self):

        self.pwd = input("Enter Your password here : ")
        self.time_btween_keys()
        self.counter_key = 0
        self.objectList.append(self.csvObjects)

        if len(self.objectList) == 8:

            print(self.test)
            writeCsv = add_info_csv(self.test)
            writeCsv.write_file()

        self.counter_key = 0

    def testing(self):

        self.pwd = input("Enter your Password : ")
        self.time_btween_keys()
        self.counter_key = 0

        return self.test[0][1:]

# class responsible for reading the and writing csv after manual stroks entry or reading from file


class add_info_csv():

    def __init__(self, mixed_info):

        self.data = mixed_info
        self.file = "learning.csv"
        self.fieldnames = ['user', 'press-0', 'release-0', 'press-1', 'release-1', 'press-2', 'release-2', 'press-3', 'release-3', 'press-4', 'release-4', 'press-5',
                           'release-5', 'press-6', 'release-6', 'press-7', 'release-7', 'press-8', 'release-8', 'press-9', 'release-9', 'press-10', 'release-10', 'press-11', 'release-11', 'press-12', 'release-12']

    def write_file(self):

        values = pd.DataFrame(self.data, columns=[self.fieldnames])
        values.to_csv('test.csv', index=False)


def main():

    print("1 Learning Mode")
    print("3 Data Entry Mode")
    print("0 exit")
    choice = input("Please choose number : ")

    if (choice == 1 or choice == '1'):

        print("Starting manual learning mode")
        manual_learning()
    elif (choice == 3 or choice == "3"):

        data_entry()
    elif(choice == 0 or choice == '0'):

        exit()
    else:

        main()


def manual_learning():

    user_id = input("Enter your user id : ")
    p = PasswordLearning(user_id)

    for _ in range(8):

        time.sleep(1)

        with keyboard.Listener(
                on_press=p.on_press,
                on_release=p.on_release) as Listen:

            p.password()
            Listen.join()

        print(_)

    main()


def data_entry():

    file_name = input("Enter File Name : ")
    data = DataTrain(file_name)
    data.main()
    main()


if __name__ == '__main__':
    main()
# for _ in range(8):
#     time.sleep(1)
#     x = threading.Thread(target=p.password)
#     x.start()
#     with keyboard.Listener(
#             on_press=p.on_press,
#             on_release=p.on_release) as Listen:
#         Listen.join()
