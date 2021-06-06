import matplotlib.pyplot as plt


class Visualization:
    """
    Visualize images and plot_attention
    """
    @staticmethod
    def process_image(img):
        """
        Preprocess the image and returns it as a np.array
        """
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224
        img[2] = img[2] * 0.225
        img[0] += 0.485
        img[1] += 0.456
        img[2] += 0.406

        return img.numpy().transpose((1, 2, 0))

    def show_image(img, title=None,fn='plot.png'):
        """Imshow for Tensor."""

        # denormalize
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224
        img[2] = img[2] * 0.225
        img[0] += 0.485
        img[1] += 0.456
        img[2] += 0.406

        image = img.cpu().numpy().transpose((1, 2, 0))

        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
        
        plt.savefig(fn)
        plt.close()  # pause a bit so that plots are updated

    def plot_attention( img, result, attention_plot,fn='plot_att.png'):
        """ Plot attention using image, caption and alphas"""
        # denormalize
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224
        img[2] = img[2] * 0.225
        img[0] += 0.485
        img[1] += 0.456
        img[2] += 0.456

        image = img.cpu().numpy().transpose((1, 2, 0))
        plt.imshow(image)
        temp_image = image

        fig = plt.figure(figsize=(15, 15))

        len_result = len(result)
        for l in range(len_result):
            temp_att = attention_plot[l].reshape(7, 7)

            ax = fig.add_subplot(len_result//3+1, 3, l + 1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att.cpu(), cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()
        plt.savefig(fn)
        plt.close()
