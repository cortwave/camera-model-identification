import telegram
import os

conf ={
    '0': {
        'key': '468286091:AAEJt1FQ3rlhcOdp0Cm_dA01WqZ3WoBg-FI',
        'chat_id': 433569393
    },
    '1': {
        'key': '405484933:AAFw1VQUUue7wspKE6xo5RGAJNxKcaKaMiM',
        'chat_id': 433569393
    }
}

class PechkaBot():
    
    def __init__(self):
        try:
            self.conf_id = os.environ['CUDA_VISIBLE_DEVICES']
            self.bot = telegram.Bot(token=conf[self.conf_id]['key'])            
            self.init_status = True
        except:
            self.init_status = False
            print('Failed to create PechkaBot')
        
    def send_message(self, txt):        
        if self.init_status:
            try:
                self.bot.send_message(conf[self.conf_id]['chat_id'], txt)
            except:
                print('PechkaBot failed to send message')
        else:
            print('PechkaBot failed to init')