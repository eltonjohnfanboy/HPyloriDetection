from PIL import Image
import random

class TripletDataset():
    def __init__(self, df, path, train=True, transform=None):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        if self.is_train:
            self.images = df.iloc[:, 1].values
            self.labels = df.iloc[:, 2].values
            self.index = df.index.values 
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        anchor_image_folder, anchor_image_name = self.images[item].split('.')
        anchor_image_path = self.path + '/' + anchor_image_folder + '/' + anchor_image_name + '.png'
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
            positive_item = random.choice(positive_list)
            positive_image_folder, positive_image_name = self.images[positive_item].split('.')
            positive_image_path = self.path + '/' + positive_image_folder + '/' + positive_image_name + '.png'
            positive_img = Image.open(positive_image_path).convert('RGB')
            #positive_img = self.images[positive_item].reshape(28, 28, 1)
            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_negative_folder, negative_image_name = self.images[negative_item].split('.')
            negative_image_path = self.path + '/' + negative_negative_folder + '/' + negative_image_name + '.png'
            negative_img = Image.open(negative_image_path).convert('RGB')
            #negative_img = self.images[negative_item].reshape(28, 28, 1)
            if self.transform!=None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)                   
                negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img, anchor_label