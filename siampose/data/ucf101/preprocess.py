import os
import selfsupmotion
import imageio
import logging
import cv2
import tqdm
import traceback

#from utils import getUCF101ClassMapping

from selfsupmotion.data.ucf101.utils import getUCF101ClassMapping
#from selfsupmotion.data.utils

def bestFitRescaleCropCenter(img, outputSize=(112, 112)):
    if not outputSize[0] == outputSize[1]:
        raise NotImplementedError('Only square crops are supported')

    # Crop based on shortest dimension of input image
    height, width, _ = img.shape
    shortestDim = min(height, width)
    startx = max(0, width // 2 - (shortestDim // 2))
    starty = max(0, height // 2 - (shortestDim // 2))
    crop = img[starty:starty + shortestDim, startx:startx + shortestDim]
    
    # Rescale based on desired dimensions
    return cv2.resize(crop, outputSize, interpolation=cv2.INTER_CUBIC)


def extractFrames(root="datasets/ucf101", outputDir='datasets/ucf101/frames_112x112', resolution=(112, 112)):
    """
    The output structure directory is:
    
    <OUTPUT DIRECTORY>
     -> split-1
        -> train
            -> class-1
                -> group-1
                    -> seq-1
                        frame_000001.jpeg
                        ...
        -> test
            -> ...
     -> split2
        -> ...
     -> split3
        -> ...
    """

    classIdsByClassNames = getUCF101ClassMapping(root)

    nbSplits = 3
    for split in range(1, nbSplits + 1):
        for partition in ['train', 'test']:
            logging.info('Processing split no.%d for partition %s' % (split, partition))
        
            # Get the list of video files for this split and partition
            filenames = []
            with open(os.path.join(root, '%slist0%d.txt' % (partition, split)), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        elems = line.split(' ')
                        filename = elems[0]
                        filenames.append(os.path.join(root, 'videos', filename))

            for filename in tqdm.tqdm(filenames):
            
                # Extract attributes for this videos
                dataType, className, groupId, seqId = os.path.splitext(os.path.basename(filename))[0].split('_')
                assert dataType == 'v'
                groupId = int(groupId[1:])
                seqId = int(seqId[1:])
                classId = classIdsByClassNames[className]
                
                outputFramesDir = os.path.join(outputDir, 'split-%d' % (split), partition, 'class-%d' % (classId), 'group-%d' % (groupId), 'seq-%d' % (seqId))
                if not os.path.exists(outputFramesDir):
                    os.makedirs(outputFramesDir)
            
                try:
                    with imageio.get_reader(filename, 'ffmpeg') as vid:
                        nframes = vid.get_meta_data()['nframes']
                        for frameId, frame in enumerate(vid):
                            frame = bestFitRescaleCropCenter(frame, resolution)
                            imageio.imwrite(os.path.join(outputFramesDir, 'frame_%06d.jpg' % (frameId)), frame, quality=95)
                except KeyboardInterrupt:
                    return
                except Exception:
                    logging.error('Unable to properly extract frames for filename: %s' % (filename))
                    traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractFrames(root="/data/sbrodeur/UCF-101", outputDir='/data/sbrodeur/UCF-101/frames_112x112', resolution=(112, 112))