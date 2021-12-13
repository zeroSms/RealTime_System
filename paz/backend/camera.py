import cv2

from ..backend.image import resize_image, convert_color_space, show_image
from ..backend.image import BGR2RGB

from head_nod_analysis import setup_variable

# ================================= パスの取得 ================================ #
path = setup_variable.path


class Camera(object):
    """Camera abstract class.
    By default this camera uses the openCV functionality.
    It can be inherited to overwrite methods in case another camera API exists.
    """

    def __init__(self, device_id=0, name='Camera'):
        # TODO load parameters from camera name. Use ``load`` method.
        self.device_id = device_id
        self.camera = None
        self.intrinsics = None
        self.distortion = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics):
        self._intrinsics = intrinsics

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def start(self):
        """ Starts capturing device

        # Returns
            Camera object.
        """
        self.camera = cv2.VideoCapture(self.device_id)
        if self.camera is None or not self.camera.isOpened():
            raise ValueError('Unable to open device', self.device_id)
        return self.camera

    def stop(self):
        """ Stops capturing device.
        """
        return self.camera.release()

    def read(self):
        """Reads camera input and returns a frame.

        # Returns
            Image array.
        """
        frame = self.camera.read()[1]
        return frame

    def is_open(self):
        """Checks if camera is open.

        # Returns
            Boolean
        """
        return self.camera.isOpened()

    def calibrate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError


# 表情のウィンドウ処理を行う
# process_windowが呼び出されるたび，face_listから表情を決定．
# face_listを初期化
face_list = {}
SCORE, NAME = 0, 1  # face_listの第1引数
def process_window():
    global face_list
    max_key = {}
    while True:
        if not face_list:
            face_list = {}
            return 'null'
        else:
            # 各キーの最大値を取得
            for k in face_list.keys():
                max_key[k] = max(face_list[k])
            print(max_key)

            face_list = {}
            return max(max_key.items(), key=lambda x: x[1])


class VideoPlayer(object):
    """Performs visualization inferences in a real-time video.

    # Properties
        image_size: List of two integers. Output size of the displayed image.
        pipeline: Function. Should take RGB image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences. Built-in pipelines can be found in
            ``paz/processing/pipelines``.

    # Methods
        run()
        record()
    """

    def __init__(self, image_size, pipeline, camera, ex_num):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera
        self.ex_num = ex_num

    def step(self):
        """ Runs the pipeline process once

        # Returns
            Inferences from ``pipeline``.
        """

        if self.camera.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')

        frame = self.camera.read()
        if frame is None:
            print('Frame: None')
            return None
        # all pipelines start with an RGB image
        frame = convert_color_space(frame, BGR2RGB)
        return self.pipeline(frame)

    def run(self):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window.
        """

        self.camera.start()
        while True:
            output = self.step()
            if output is None:
                continue
            image = resize_image(output['image'], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            if cv2.waitKey(1) & 0xFF == 13:  # 13 = Enter
                break
        self.camera.stop()
        cv2.destroyAllWindows()

    def record(self, fps=20, fourCC='DIVX'):
        """Opens camera and records continuous inference using ``pipeline``.

        # Arguments
            name: String. Video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """
        global face_list
        self.camera.start()
        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(path + '/face/video/video'+self.ex_num+'.avi', fourCC, fps, self.image_size)
        while True:
            output = self.step()
            if output is None:
                continue

            # 表情スコアの出力(辞書型)
            if len(output['boxes2D']) != 0:
                box2D = output['boxes2D'][0]
                # print(output)
                face_name = box2D.class_name
                if face_name not in face_list:
                    face_list[face_name] = []
                face_list[face_name].append(box2D.score)
            # print(face_list)

            image = resize_image(output['image'], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == 13:  # 13 = Enter
                break

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()
