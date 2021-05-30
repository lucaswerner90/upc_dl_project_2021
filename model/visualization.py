import matplotlib.pyplot as plt


class Visualization:
    """
    Visualize images and plot_attention
    """

    def __init__(self, img):
        self.img = img

    def show_image(self, img, title=None):
        """Imshow for Tensor."""

        # denormalize
        self.img[0] = self.img[0] * 0.229
        self.img[1] = self.img[1] * 0.224
        self.img[2] = self.img[2] * 0.225
        self.img[0] += 0.485
        self.img[1] += 0.456
        self.img[2] += 0.406

        image = self.img.numpy().transpose((1, 2, 0))

        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_attention(self, img, result, attention_plot):
        """ Plot attention using image, caption and alphas"""
        # denormalize
        self.img[0] = self.img[0] * 0.229
        self.img[1] = self.img[1] * 0.224
        self.img[2] = self.img[2] * 0.225
        self.img[0] += 0.485
        self.img[1] += 0.456
        self.img[2] += 0.456

        image = self.img.numpy().transpose((1, 2, 0))
        temp_image = image

        fig = plt.figure(figsize=(15, 15))

        len_result = len(self.result)
        for l in range(len_result):
            temp_att = self.attention_plot[l].reshape(7, 7)

            ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
            ax.set_title(self.result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    # TODO:integrate the following function
    # def get_caps_from(features_tensors):
    #     # generate the caption
    #     model.eval()
    #     with torch.no_grad():
    #         features = model.encoder(features_tensors.to(device))
    #         caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
    #         caption = ' '.join(caps)
    #         show_image(features_tensors[0], title=caption)
    #
    #     return caps, alphas


#
# #show any 1
# dataiter = iter(data_loader)
# images,_ = next(dataiter)
#
# img = images[0].detach().clone()
# img1 = images[0].detach().clone()
# caps,alphas = get_caps_from(img.unsqueeze(0))
#
# plot_attention(img1, caps, alphas)
