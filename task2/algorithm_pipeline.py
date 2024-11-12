import re
import os
import rasterio
import numpy as np
import torch
from kornia.feature import LoFTR
import imageio

class ImageMatcher():
    def __init__(self, path1, path2, bands=["04", "03", "08"], target_size=(732, 732)):
        """
        Initialize the ImageMatcher class.
        
        Parameters:
        path1 (str): Path to the first image.
        path2 (str): Path to the second image.
        bands (list): List of band names to use for the images.
        target_size (tuple): Desired size of the images after preprocessing.
        """
        self.path1 = path1
        self.path2 = path2
        self.bands = bands
        self.target_size = target_size
        self.matcher = LoFTR(pretrained='outdoor')
        self.matcher.eval()

    def _imread_3bands(self, path, bands):
        """
        Load and preprocess a 3-band satellite image.
        
        Parameters:
        path (str): Base path to the image files.
        bands (list): List of band names to load.
        
        Returns:
        numpy.ndarray: 3D array containing the preprocessed image.
        """
        channels = []
        for band in bands:
            with rasterio.open(f'{path}{band}.jp2', "r", driver="JP2OpenJPEG") as src:
                channel = src.read(1).astype(float)
                q1 = np.percentile(channel, 25)
                q3 = np.percentile(channel, 75)
                iqr = q3 - q1
                channel /= q3 + 1.5 * iqr
                channel_8bit = (np.clip(channel, 0, 1) * 255).astype(np.uint8)
                channels.append(channel_8bit)
        return np.stack(channels, axis=-1)

    def _divide_into_pieces(self, image_array, save_path, width, height):
        """
        Divide a 3D image array into smaller pieces and save them as individual PNG files.
        
        Parameters:
        image_array (numpy.ndarray): 3D array containing the image data.
        save_path (str): Path to the directory where the image pieces will be saved.
        width (int): Width of each image piece.
        height (int): Height of each image piece.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(f'{save_path}/images', exist_ok=True)
            print('Data directory created.')

        full_height, full_width = image_array.shape[:2]
        for j in range(0, full_height // height):
            for i in range(0, full_width // width):
                window = image_array[
                    j * height:(j + 1) * height,
                    i * width:(i + 1) * width,
                    :3
                ]
                piece_name = f'piece_{j}_{i}.png'
                imageio.imwrite(f'{save_path}/{piece_name}', window)

    def _load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image.
        
        Parameters:
        image_path (str): Path to the input image.
        
        Returns:
        numpy.ndarray: Preprocessed image as a numpy array.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        
        return image

    def _match_pieces(self, piece1_path, piece2_path):
        """
        Match keypoints between two image pieces using the LoFTR model.
        
        Parameters:
        piece1_path (str): Path to the first image piece.
        piece2_path (str): Path to the second image piece.
        
        Returns:
        numpy.ndarray: First image piece.
        numpy.ndarray: Second image piece.
        numpy.ndarray: Keypoints from the first image piece.
        numpy.ndarray: Keypoints from the second image piece.
        numpy.ndarray: Confidence scores for the matches.
        """
        img1 = self._load_and_preprocess_image(piece1_path)
        img2 = self._load_and_preprocess_image(piece2_path)
        
        img1_tensor = torch.from_numpy(img1)[None][None]
        img2_tensor = torch.from_numpy(img2)[None][None]
        
        with torch.no_grad():
            data = {'image0': img1_tensor, 'image1': img2_tensor}
            matches = self.matcher(data)
    
            mkpts0 = matches['keypoints0'].cpu().numpy()
            mkpts1 = matches['keypoints1'].cpu().numpy()
            confidence = matches['confidence'].cpu().numpy()
            
        return img1, img2, mkpts0, mkpts1, confidence
    
    def get_matching(self):
        """
        Perform image matching between the two input images.
        
        Returns:
        dict: A dictionary mapping image piece coordinates to the matching results.
        """
        # Load and preprocess the full-size images
        im1 = self._imread_3bands(self.path1, self.bands)
        im2 = self._imread_3bands(self.path2, self.bands)
        
        # Determine the names of the image piece directories
        piece1_path = self.path1.split('/')[-3]
        piece2_path = self.path2.split('/')[-3]
        
        # Divide the full-size images into smaller pieces
        self._divide_into_pieces(im1, piece1_path, *self.target_size)
        self._divide_into_pieces(im2, piece2_path, *self.target_size)
        
        # Find the paths of the individual image pieces
        pattern = re.compile(r'_(\d+)_(\d+)\.png$')
        def get_pieces(dir):
            pieces = {}
            for filename in os.listdir(dir):
                match = pattern.search(filename)
                if match:
                    i = int(match.group(1))
                    j = int(match.group(2))
                    pieces[(i, j)] = dir+'/'+filename    
            return pieces
        
        pieces1 = get_pieces(piece1_path)
        pieces2 = get_pieces(piece2_path)
        
        # Match the corresponding image pieces
        pieces_matchings = {}
        for p in pieces1:
            pieces_matchings[p] = self._match_pieces(pieces1[p], pieces2[p])
            print(pieces1[p].split('/')[-1], " is analyzed.")
        return pieces_matchings
    

if __name__ == "main":
    im1_path = "Sentinel2/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE/GRANULE/L1C_T36UYA_A003350_20160212T084510/IMG_DATA/T36UYA_20160212T084052_B"
    im2_path = "Sentinel2/S2B_MSIL1C_20190412T083609_N0207_R064_T36UYA_20190412T122445/S2B_MSIL1C_20190412T083609_N0207_R064_T36UYA_20190412T122445.SAFE/GRANULE/L1C_T36UYA_A010958_20190412T084433/IMG_DATA/T36UYA_20190412T083609_B"
    image_matcher = ImageMatcher(im1_path, im2_path)
    results = image_matcher.get_matching()
    np.save('results.npy', results)
