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
face_list = [[], []]
SCORE, NAME = 0, 1  # face_listの第1引数
def process_window():
    global face_list
    while True:
        if len(face_list[SCORE]) == 0:
            face_list = [[], []]
            return ["null"]
        else:
            max_name = [face_list[NAME][i] for i, v in enumerate(face_list[SCORE]) if v == max(face_list[SCORE])]
            face_list = [[], []]
            return max_name


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

    def __init__(self, image_size, pipeline, camera):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera

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

    def record(self, name=path+'/video/video.avi', fps=20, fourCC='DIVX'):
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
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)
        while True:
            output = self.step()
            if output is None:
                continue
            # 表情スコアの出力
            if len(output['boxes2D']) != 0:
                box2D = output['boxes2D'][0]
                face_list[SCORE].append(box2D.score)
                face_list[NAME].append(box2D.class_name)
            print(face_list)
            image = resize_image(output['image'], tuple(self.image_size))
            show_image(image, 'inference', wait=False)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == 13:  # 13 = Enter
                break

        self.camera.stop()
        writer.release()
        cv2.destroyAllWindows()
