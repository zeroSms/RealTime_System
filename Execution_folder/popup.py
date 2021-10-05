import time
import winsound

from plyer import notification
import random

action_list = ['うなずき 大', 'うなずき 小', '首振り 大', '首振り 小']
action_num = 0
TimeOver = 4 * 10


def wait_10():
    time.sleep(10)


class PopUp:
    def Action(self, action_name):
        if 'うなずき' in action_name:   push_button = 1
        else:   push_button = 2
        notification.notify(
            title=action_name,
            message=str(push_button),
            app_icon="python.ico",
            timeout=5
        )

    def End(self):
        notification.notify(
            title="終了の通知",
            message='10分経過しました．\n動画を停止してください．',
            app_icon="python.ico",
            timeout=5
        )


# ================================= メイン関数　実行 ================================ #
if __name__ == '__main__':
    rand_action = random.sample(action_list, k=4)
    popup = PopUp()

    for i in range(4):
        action_num += 1
        winsound.Beep(800, 100)
        popup.Action(rand_action[i])
        if action_num == TimeOver:  break
        wait_10()

    winsound.Beep(400, 100)
    popup.End()
