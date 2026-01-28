---
layout: default
title: Chess Image Generation
---

<h1 style="text-align:center"><strong>Chess Image Generation: Synthetic to Real</strong></h1>

<p style="text-align:center">
 
  <strong>
    <a href="https://www.linkedin.com/in/yoav-baider/">Yoav Baider Klein</a> · 
    <a href="https://www.linkedin.com/in/maor-haak/">Maor Haak</a> · 
    <a href="https://www.linkedin.com/in/guy-kouchly/">Guy Kouchly</a>
  </strong>
</p>



<div style="text-align:center;">
  <a class="button" href="/assets/Final Report.pdf" style="margin-right:4px;">
    <img src="{{ '/assets/paper_icon.png' | relative_url }}" alt="Paper" style="height: 60px; vertical-align: middle;">
  </a>
  <a class="button" href="https://github.com/BHK-ImaGen" style="margin-right:4px;">
    <img src="{{ '/assets/github_icon.png' | relative_url }}" alt="Code" style="height: 60px; vertical-align: middle;">
  </a>
</div>

<br>


![Image]({{ '/assets/teaser_image.jpg' | relative_url }})




## **Abstract**


This project addresses the problem of synthetic to real image translation for chessboard rendering, with the goal of transforming synthetic chessboard images into matching
realistic looking photographs. We propose a (cGAN) that performs supervised image to
image translation from the synthetic renders to the real image domain. Results show that
the proposed method successfully transforms the synthetic inputs to realistic looking
images while maintaining accurate geometric consistency.

---

## **How It Works**

Our model is a supervised conditional GAN based on the pix2pix framework that translates synthetic chessboard renderings into realistic chessboard images. A U-Net–based generator with skip connections maps 512×512 RGB synthetic inputs to RGB outputs using strided convolutions, mirrored upsampling, and a tanh activation, while adversarial learning is provided by a PatchGAN discriminator operating on paired input–output images. Training is fully supervised using paired synthetic–real data, with joint optimization of the generator and discriminator using adversarial, reconstruction, and perceptual losses, stabilized through asymmetric learning rates and an R1 gradient penalty. The model operates purely on image data, with normalization to [−1, 1], and is evaluated qualitatively through periodic checkpointing and visual inspection.

<div style="text-align:center">

  <img src="/assets/U-Net.png" alt="Image 1" width="45%" style="margin: 0 0px;">
  <img src="/assets/PatchGAN.png" alt="Image 2" width="45%" style="margin: 0 0px;">

</div>
<br>


---

## **Data Collection Methods**

Our data collection pipeline emphasizes scalability, automation, and visual realism. In addition to real chessboard images labeled with FEN, we expand the dataset using PGN game data through an automatic labeling and alignment process built on the [Fenify 3D](https://github.com/notnil/fenify-3D) framework, which matches video frames to reconstructed game states while accounting for board symmetries and temporal consistency. Dataset quality is improved by filtering frames with hand occlusions using [MediaPipe](https://github.com/google-ai-edge/mediapipe), and diversity is increased through board-level recomposition that generates new realistic boards from real square tiles. All synthetic and real images are preprocessed into a unified 512×512 RGB format with normalization, ensuring consistent inputs for training and evaluation.

<div style="text-align:center">

  <img src="/assets/Image2FEN.png" alt="Image to FEN" width="90%" style="margin: 0 0px;">

</div>
<br>

---

## **References**

\[1\] Isola et al., ["Image-to-Image Translation with Conditional Adversarial Networks"][1], CVPR 2017. <br>
\[2\] [Fenify repository / tool][2]. <br>
\[3\] [python-chess: a chess library for Python][3] <br>
\[4\] [Google MediaPipe Hands repository][4].


[1]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
[2]: https://github.com/notnil/fenify-3D
[3]: https://python-chess.readthedocs.io/en/latest/

[4]: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
