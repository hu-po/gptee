import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import base64
import requests
import io
from scipy.io.wavfile import write
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import os

class VisionTTSNode(Node):
    def __init__(self):
        super().__init__('vision_tts_node')
        
        # Declare and get parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('openai_api_key', os.environ.get('OPENAI_API_KEY')),
                ('vision_model', 'gpt-4-vision-preview'),
                ('tts_model', 'tts-1'),
                ('stt_model', 'whisper-1'),
                ('voice', 'echo'),
                ('camera_wait_ms', 2056),
                ('max_tokens_vision', 32),
                ('audio_record_seconds', 6),
                ('audio_sample_rate', 22100),
                ('audio_channels', 1),
                ('audio_output_path', '/tmp/gpt_audio.wav')
            ]
        )
        self.api_key = self.get_parameter('openai_api_key').value
        self.vision_model = self.get_parameter('vision_model').value
        self.tts_model = self.get_parameter('tts_model').value
        self.stt_model = self.get_parameter('stt_model').value
        self.voice = self.get_parameter('voice').value
        self.camera_wait_ms = self.get_parameter('camera_wait_ms').value
        self.max_tokens_vision = self.get_parameter('max_tokens_vision').value
        self.audio_record_seconds = self.get_parameter('audio_record_seconds').value
        self.audio_sample_rate = self.get_parameter('audio_sample_rate').value
        self.audio_channels = self.get_parameter('audio_channels').value
        self.audio_output_path = self.get_parameter('audio_output_path').value

        # Initialize a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscription to the image topic
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to camera image topic.')

        # Initialize the OpenAI client
        self.client = None  # This should be replaced with the actual client initialization, if available

    def image_callback(self, msg):
        try:
            # Convert ROS2 image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the image and perform API calls, etc.
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error('Failed to process image: %r' % (e,))

    def process_image(self, cv_image):
        # Encode the image as base64
        base64_image = self.encode_image_to_base64(cv_image)

        # Record audio
        self.record_audio()

        # Transcribe audio to text
        prompt = self.transcribe_audio()

        # Send to vision model
        reply = self.vision(prompt, base64_image)

        # Play audio response
        self.text2speech(reply)

    def encode_image_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

    def record_audio(self):
        self.get_logger().info(f'Recording for {self.audio_record_seconds} seconds.')
        audio_data = sd.rec(
            int(self.audio_record_seconds * self.audio_sample_rate),
            samplerate=self.audio_sample_rate,
            channels=self.audio_channels,
        )
        sd.wait()  # Wait until recording is finished
        write(self.audio_output_path, self.audio_sample_rate, audio_data)  # Save as WAV file

    def transcribe_audio(self, audio_path=None):
        if audio_path is None:
            audio_path = self.audio_output_path

        # This should be replaced with the actual transcription code using your OpenAI client
        transcript = 'This is a dummy transcript.'
        return transcript

    def vision(self, prompt, base64_image):
        # This should be replaced with the actual vision API call using your OpenAI client
        response = 'This is a dummy response from the vision model.'
        return response

    def text2speech(self, text):
        # This should be replaced with the actual TTS API call using
        pass

def main():
    rclpy.init()
    node = VisionTTSNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()