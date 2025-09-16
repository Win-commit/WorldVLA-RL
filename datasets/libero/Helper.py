import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import numpy as np
from typing import List, Dict, Tuple
class ImageSimilarityCalculator:
    def __init__(self):
        self.orb = cv2.ORB_create(
            edgeThreshold=0,
            fastThreshold=40
        )

        # self.orb = cv2.ORB(
        #     edgeThreshold=0,
        #     fastThreshold=40
        # )

    def calculate_mse(self, img1, img2):
        """
        Calculate MSE between two images.

        Args:
            img1 (numpy.ndarray): The first image.
            img2 (numpy.ndarray): The second image.

        Returns:
            float: The MSE value between the two images.
        """
        return mean_squared_error(img1, img2)

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images.

        Args:
            img1 (numpy.ndarray): The first image.
            img2 (numpy.ndarray): The second image.

        Returns:
            float: The SSIM value between the two images.
        """
        return ssim(img1, img2, channel_axis=-1)

    def calculate_orb_similarity(self, img1, img2, save_image=False):
        """
        Calculate similarity between two images using ORB feature matching.

        Args:
            img1_path (str): Path to the first image.
            img2_path (str): Path to the second image.

        Returns:
            float: The ratio of matched points to total points as a measure of similarity.
        """
        # Read images
        # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        # img1 = Image.fromarray(img1).convert('L')
        # img2 = Image.fromarray(img2).convert('L')

        img1_original = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_original = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        img1 = cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0, None  # No keypoints detected

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # print("matches:",matches)

        img_matches = None
        if save_image:
            # 绘制前N个匹配项
            N = 500  # 选择前N个最好的匹配项
            # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
            img_matches = cv2.drawMatches(img1_original, kp1,
                                          img2_original, kp2,
                                          matches[:N], None, flags=2)

        # Calculate the ratio of matched points to total points
        num_matches = len(matches)
        num_keypoints = min(len(kp1), len(kp2))
        # print("num_matches:",num_matches)
        # print("num_keypoints:",num_keypoints)

        similarity_ratio = num_matches / num_keypoints if num_keypoints > 0 else 0.0

        return similarity_ratio, img_matches

    def calculate_visual_reward(self, img1, img2, save_image=False):
        mse_value = self.calculate_mse(img1, img2)
        ssim_value = self.calculate_ssim(img1, img2)
        orb_similarity, img_matches = self.calculate_orb_similarity(img1,
                                                                    img2,
                                                                    save_image=save_image
                                                                    )
        r_mse = np.exp(-0.01 * mse_value)
        r_ssim = np.exp(1. * (ssim_value - 1.))
        r_orb = np.exp(1. * (orb_similarity - 1.))
        return r_mse, r_ssim, r_orb, img_matches




class RewardCalculator():
    def __init__(self, episode_states:np.ndarray, episode_actions:np.ndarray, images:np.ndarray, wrist_images:np.ndarray, **kwargs):
        '''
        episode_states:一个轨迹的每个时间步的状态,N*D1
        episode_actions:一个轨迹的每个时间步的动作 N*D2
        images:第三人称视图
        wrist_images:机械臂视图
        '''
        self.joint_pos = episode_states[:,:-2]
        self.gripper_action = (episode_states[:,-1:]>np.mean(episode_states[:,-1:])).astype(np.float32)
        self.actions = episode_actions
        self.images = images
        self.wrist_images = wrist_images
    

    def _is_stopped(self, demo: List[Dict], i: int, obs: Dict, stopped_buffer: int, delta=0.1) -> bool:
        '''
        根据启发式规则判断机械臂在时刻i是否停止
        '''
        next_is_not_final = i == (len(demo) - 2)
        gripper_state_no_change = (
                i < (len(demo) - 2) and
                (obs["gripper_open"] == demo[i + 1]["gripper_open"] and
                obs["gripper_open"] == demo[i - 1]["gripper_open"] and
                demo[i - 2]["gripper_open"] == demo[i - 1]["gripper_open"]))
        small_delta = np.allclose(obs["joint_velocities"], 0, atol=delta)
        stopped = (stopped_buffer <= 0 and small_delta and
                (not next_is_not_final) and gripper_state_no_change)
        return stopped
    
    def keypoint_discovery(self, demo: List[Dict], stopping_delta=0.1, method='heuristic') -> List[int]:
        '''
        返回一个episode中关键帧的index
        '''
        episode_keypoints = []
        if method == 'heuristic':
            prev_gripper_open = demo[0]["gripper_open"]
            stopped_buffer = 0
            for i, obs in enumerate(demo):
                stopped = self._is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
                stopped_buffer = 4 if stopped else stopped_buffer - 1
                # If change in gripper, or end of episode.
                last = i == (len(demo) - 1)
                if i != 0 and (obs["gripper_open"] != prev_gripper_open or
                            last or stopped):
                    episode_keypoints.append(i)
                prev_gripper_open = obs["gripper_open"]
            if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                    episode_keypoints[-2]:
                episode_keypoints.pop(-2)
            return episode_keypoints

        elif method == 'random':
            # Randomly select keypoints.
            episode_keypoints = np.random.choice(
                range(len(demo)),
                size=20,
                replace=False)
            episode_keypoints.sort()
            return episode_keypoints.tolist()

        elif method == 'fixed_interval':
            # Fixed interval.
            episode_keypoints = []
            segment_length = len(demo) // 20
            for i in range(0, len(demo), segment_length):
                episode_keypoints.append(i)
            return episode_keypoints

        else:
            raise NotImplementedError

    def remove_elements_with_small_interval(self, lst:List, threshold:int):
        i = 1
        while i < len(lst):
            if lst[i] - lst[i - 1] < threshold:
                lst.pop(i)
            else:
                i += 1
        return lst
    
    def get_keypoints_reward(self, 
                         stopping_delta:float = 0.001
                         ) ->  Tuple[np.ndarray, List, np.ndarray]:
        #关节速度
        diff_arr = np.zeros_like(self.joint_pos)
        diff_arr[1:] = np.diff(self.joint_pos, axis=0)
        #关节加速度
        diff_arr_2 = np.zeros_like(self.joint_pos)
        diff_arr_2[1:] = np.diff(diff_arr, axis=0)


        error_action = np.zeros_like(self.actions)
        error_action[1:] = np.diff(self.actions, axis=0)
        error_action_2 = np.zeros_like(self.actions)
        error_action_2[1:] = np.diff(error_action, axis=0)

        demo = []
        for i in range(len(self.gripper_action)):
            demo.append({
                "joint_velocities": diff_arr[i],
                "gripper_open": self.gripper_action[i],
            })
        episode_keypoints = self.keypoint_discovery(demo, stopping_delta=stopping_delta)  
        while episode_keypoints[0] < 5:
            del episode_keypoints[0]

        episode_keypoints = self.remove_elements_with_small_interval(episode_keypoints, 10)

        print("episode_keypoints:", episode_keypoints)
        if episode_keypoints[-1]!=len(self.actions)-1: episode_keypoints.append(len(self.actions)-1)    

        # reward calculation by using keypoints
        calculator = ImageSimilarityCalculator()
        # Calculate MSE
        reward_list = []

        for i in range(np.shape(self.joint_pos)[0]):
            reward_subgoal = 0.
            for subgoal_ind in episode_keypoints:
                reward_subgoal += 1 / len(episode_keypoints)
                if i <= subgoal_ind:
                    subgoal_id = subgoal_ind
                    break
            reward_success = 1.

            save_image = False
            mse_value, ssim_value, orb_similarity, img_matches = calculator.calculate_visual_reward(self.images[i],
                                                                                                    self.images[subgoal_id],
                                                                                                    save_image=save_image)

            # if img_matches is not None: cv2.imwrite('my_test_keypoint/matches_result_{}.png'.format(str(i)), img_matches)
            mse_value_gripper, ssim_value_gripper, orb_similarity_gripper, img_matches_gripper = calculator.calculate_visual_reward(
                self.wrist_images[i],
                self.wrist_images[subgoal_id],
                save_image=save_image)

            # if img_matches_gripper is not None:
            #     print("str(i)", str(i))
            #     cv2.imwrite('my_test_keypoint/matches_result_{}_gripper.png'.format(str(i)), img_matches_gripper)

            error_joint = np.linalg.norm(self.joint_pos[i] - self.joint_pos[subgoal_id])
            error_joint = np.exp(-1. * error_joint)

            # smoothing
            error_joint_acc = np.sum(np.abs(diff_arr_2[i]))
            error_joint_vel = np.sum(np.abs(diff_arr[i]))

            error_action_vel = np.sum(np.abs(error_action[i]))
            error_action_acc = np.sum(np.abs(error_action_2[i]))

            reward_single = np.array([
                # image goal tracking
                mse_value,
                ssim_value,
                orb_similarity,
                mse_value_gripper,
                ssim_value_gripper,
                orb_similarity_gripper,

                # prop. goal tracking
                error_joint,

                # auxiliary rewards
                error_joint_vel,
                error_joint_acc,
                error_action_vel,
                error_action_acc,

                # sub-goal progress
                reward_subgoal,
                reward_success
            ])

            reward_weight = np.array([
                # image goal tracking
                1 / len(reward_single),
                1 / len(reward_single),
                1 / len(reward_single),
                1 / len(reward_single),
                1 / len(reward_single),
                1 / len(reward_single),

                # prop. goal tracking
                1 / len(reward_single),

                # auxiliary rewards
                -0.1 / len(reward_single) * 10 ** (1),
                -0.1 / len(reward_single) * 10 ** (1),
                -0.01 / len(reward_single) * 10 ** (1),
                -0.01 / len(reward_single) * 10 ** (1),

                # sub-goal progress
                1 / len(reward_single),
                1 / len(reward_single)
            ])
            reward_total = (reward_weight * reward_single) / 10
            reward_list.append(np.concatenate((reward_total, np.sum(reward_total, keepdims=True))))

        reward_np = np.array(reward_list)
        return self.scale_reward(reward_np), episode_keypoints, diff_arr
    
    def scale_reward(self, reward:np.ndarray):
        reward_min, reward_max = np.min(reward,axis=0), np.max(reward, axis=0)
        reward = (reward - reward_min)/( reward_max - reward_min + 10**(-6))
        reward = 0.1 * reward
        return reward
        