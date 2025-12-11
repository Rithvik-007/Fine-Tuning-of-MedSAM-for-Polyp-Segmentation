\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{titlesec}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{geometry}
\geometry{margin=1in}

\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}

\title{\textbf{Domain-Specific Fine-Tuning of MedSAM for Polyp Segmentation}}
\author{Ashwin Ravichandran \and Rithvik Pranao Nagaraj \\ CS 747 -- Deep Learning, Fall 2025}
\date{}

\begin{document}
\maketitle

\section{Introduction}
This project investigates how well the \textbf{MedSAM} medical foundation model generalizes to an under-represented modality (endoscopic polyp segmentation) and whether \textbf{domain-specific fine-tuning} improves its performance. We evaluate three models:

\begin{itemize}[itemsep=4pt]
    \item \textbf{Original MedSAM (Generalist)} -- pretrained on 1.57M medical image--mask pairs.
    \item \textbf{Fine-Tuned MedSAM (Specialist)} -- adapted specifically to the Kvasir-SEG dataset.
    \item \textbf{MedSamLike (Scratch Model)} -- a lighter ResNet-based baseline trained only on Kvasir.
\end{itemize}

The primary goal is to test whether a foundation model pretrained on many modalities can outperform or match specialized training, and how much fine-tuning improves its predictions on a narrow domain.

\section{Dataset: Kvasir-SEG}
The Kvasir-SEG dataset contains:
\begin{itemize}[itemsep=3pt]
    \item 1,000 RGB endoscopy images,
    \item pixel-wise binary masks (polyp vs. background),
    \item train/validation split: 880 / 120 images.
\end{itemize}

All images were resized to $512\times512$ for the scratch model and $1024\times1024$ for MedSAM, following architectural requirements. Bounding boxes were computed from ground-truth masks for prompt generation.

\section{Model Architectures}

\subsection{MedSAM (Generalist)}
The original MedSAM architecture includes:
\begin{itemize}[itemsep=3pt]
    \item ViT-B image encoder (from Segment Anything Model),
    \item prompt encoder (for bounding boxes),
    \item mask decoder with upsampling layers.
\end{itemize}
It was evaluated \textbf{zero-shot}, without training on Kvasir-SEG.

\subsection{Fine-Tuned MedSAM}
Fine-tuning followed the MedSAM paper:
\begin{itemize}[itemsep=3pt]
    \item prompt encoder frozen,
    \item image encoder + mask decoder updated,
    \item trained with Dice + BCE loss on the Kvasir train set.
\end{itemize}

\subsection{MedSamLike (Scratch Model)}
A smaller custom segmentation model built with:
\begin{itemize}[itemsep=2pt]
    \item ResNet-based encoder,
    \item lightweight convolutional decoder,
    \item bounding-box prompt embedding.
\end{itemize}
Trained from scratch for 40--50 epochs.

\section{Training Pipeline}
\subsection{Loss Functions}
We used:
\begin{itemize}[itemsep=3pt]
    \item \textbf{Dice Loss} (handles class imbalance),
    \item \textbf{Binary Cross-Entropy Loss}.
\end{itemize}

\subsection{Evaluation Metric}
All models were evaluated using the \textbf{Dice coefficient}:
\[
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}
\]
where $P$ is the predicted mask and $G$ is the ground-truth mask.

\section{Results}

\subsection{Dice Score Comparison}
\begin{center}
\begin{tabular}{|l|c|}
\hline
\textbf{Model} & \textbf{Mean Dice Score} \\
\hline
MedSamLike (Scratch) & 0.8640 \\
MedSAM (Generalist) & 0.9081 \\
MedSAM (Fine-Tuned) & \textbf{0.9448} \\
\hline
\end{tabular}
\end{center}

\subsection{Interpretation}
\begin{itemize}
    \item The foundation model (MedSAM) performs strongly even zero-shot.
    \item Fine-tuning gives a substantial improvement (+0.0367 Dice over generalist).
    \item Training from scratch on a small dataset performs reasonably but cannot match foundation models.
\end{itemize}

\section{Visual Comparisons}
Side-by-side comparison grids were generated showing:
\begin{itemize}[itemsep=3pt]
    \item original image,
    \item ground truth,
    \item scratch model prediction,
    \item MedSAM (generalist) prediction,
    \item MedSAM (fine-tuned) prediction.
\end{itemize}

Example file locations:
\begin{itemize}
    \item \texttt{work\_dir/vis\_comparison/compare\_0001.png}
\end{itemize}

\section{Reproducibility}

\subsection{Training Scratch Model}
\begin{verbatim}
python -m src.train_medsam_kvasir --data_root data/Kvasir-SEG --device cuda:0
\end{verbatim}

\subsection{Evaluating Generalist MedSAM}
\begin{verbatim}
python -m src.eval_original_medsam_kvasir --checkpoint work_dir/MedSAM/medsam_vit_b.pth
\end{verbatim}

\subsection{Fine-Tuning MedSAM}
\begin{verbatim}
python -m src.train_finetune_medsam --checkpoint work_dir/MedSAM/medsam_vit_b.pth
\end{verbatim}

\subsection{Dice Comparison Plot}
\begin{verbatim}
python -m src.plot_from_checkpoints --device cuda:0
\end{verbatim}

\section{Conclusion}
This project demonstrates that:
\begin{itemize}[itemsep=3pt]
    \item MedSAM generalizes well across unseen modalities,
    \item fine-tuning on a small dataset significantly boosts performance,
    \item scratch-trained models struggle to match foundation models without large datasets.
\end{itemize}

Fine-tuned MedSAM provides the strongest segmentation quality, supporting the importance of specialization even for powerful foundation models.

\section*{Authors}
\textbf{Ashwin Ravichandran} \\
\textbf{Rithvik Pranao Nagaraj} \\
CS 747 -- Deep Learning \\
Instructor: Prof. Daniel Barbara

\end{document}
