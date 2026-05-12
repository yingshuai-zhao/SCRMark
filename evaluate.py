from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import re
from wm_models.seg_model import U2NET
from wm_models.msg_ed import Msg_ED
from utils import *
from seg import obtain_wm_blocks

setup_seed(30)

def decode(decoder, noised_images, seg_model, name):
    """
    Decode images or noised images
    """
    with torch.no_grad():
        noised_blocks = obtain_wm_blocks(noised_images, seg_model, name)
        decode_messages = []
        n = 1
        for _ in range(0, len(noised_blocks), n):
            m, _ = decoder(noised_blocks[_:_+n])
            decode_messages.append(m)
        decode_messages = torch.vstack(decode_messages)
    
    return decode_messages

def message_evaluate(messages, decode_messages, mode='mean'):
    """
    mode: mean, min, fusion
    return the bit error rate between messages and decode_messages
    """
    error_rate_bit = decoded_message_error_rate_bit_batch(messages, decode_messages, mode)

    return error_rate_bit


def search_dir_file(rootdir):
    """
    return image file path in natural sorting order
    """
    allfile = []

    names = []
    
    list_dir = os.listdir(rootdir)
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', s)]
    
    for dir_or_file in sorted(list_dir, key=natural_sort_key):
        filePath = os.path.join(rootdir, dir_or_file)
        if os.path.isfile(filePath):
            if os.path.basename(filePath).lower().endswith(('.jpg', '.png', '.jpeg')):
                allfile.append(filePath)
                names.append(dir_or_file)
        elif os.path.isdir(filePath):
            allfile.extend(search_dir_file(filePath))
        else:
            print('not file and dir ' + os.path.basename(filePath))
    
    return allfile, names


if __name__ == '__main__':
    # select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    wm_dec_path = './weights/wm_dec.pt'
    seg_path = './weights/seg.pt'
    message_length = 64

    wm_model = Msg_ED(message_length)

    wm_model.decoder.load_state_dict(torch.load(wm_dec_path, map_location=torch.device('cpu')))
    wm_model.decoder.to(device)
    wm_model.decoder.eval()


    seg_model = U2NET(mode='eval')
    checkpoint = torch.load(seg_path, map_location='cpu')
    seg_model.load_state_dict(checkpoint['state_dict'])
    seg_model.to(device)
    seg_model.eval()


    captured_path = 'captured_path'

    print("================Decoding================")
    messages_all = torch.load(captured_path + '/msg.pth', map_location=torch.device('cpu'))
    messages_all.to(device)
    noise_loader, names = search_dir_file(captured_path)
    print("================Test:==============")
    bit_accuracy_min = []
    bit_accuracy_mean = []
    bit_accuracy = []

    messages_all_re = []
    
    for batch_idx, batch_data in enumerate(tqdm(noise_loader)):

        curr_img_path = noise_loader[batch_idx:batch_idx+1]
        name = names[batch_idx:batch_idx+1]

        n1 = name[0].index('.')
        num = int(name[0][:n1])

        ori_messages = messages_all[num-1:num].to(device)

        # decode
        captured_images = [transform(Image.open(idx).convert('RGB')) for idx in noise_loader[batch_idx:batch_idx+1]]
        captured_images = torch.stack(captured_images, dim=0)
        captured_images = captured_images.to(device)

        decode_messages = decode(wm_model.decoder, captured_images, seg_model, str(num))

        # evaluate
        decode_messages = messgae_fusion(decode_messages)

        bit_error = message_evaluate(ori_messages, decode_messages, mode='fusion')

        print('\nOri_Message: ', ori_messages, '\nPre_Message: ', (torch.nn.Sigmoid()(decode_messages[0]) > 0.5).int(), '\nBit_ACC: {}'.format(1-bit_error))

        bit_accuracy.append(1-bit_error)

    print('Bit_ACC_Mean: {}'.format(np.mean(bit_accuracy)))