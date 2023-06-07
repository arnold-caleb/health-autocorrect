## Title: Enhancing Radiology Reporting with Image-Text Grounding Models

# Objective:
- Develop an image-text grounding system to assist radiologists in creating accurate and meaningful descriptions of medical images by identifying potential discrepancies, suggesting alternative terms, and providing real-time feedback.

# Background:
- Radiologists often need to describe and interpret complex medical images, which can be prone to errors and misinterpretations. Leveraging image-text grounding models to improve radiological reporting can help minimize mistakes and enhance overall diagnostic accuracy.

# Proposed System Features:
1. Real-time Feedback: Flag words that do not match the image content, allowing radiologists to reconsider their descriptions before finalizing the report.
2. Automated Suggestions: Offer alternative suggestions for flagged words based on the image content.
3. Confidence Scores: Assign confidence scores to the grounding of specific terms to gauge the reliability of generated suggestions.
4. Visualizations: Provide visualizations highlighting regions of the image corresponding to specific terms, aiding radiologists in understanding the grounding process.
5. Learning from Radiologists: Incorporate expert feedback to improve grounding accuracy and adapt suggestions accordingly.
6. Integration with EHRs: Streamline the radiology report generation process by enabling seamless access to relevant patient information.
7. Collaboration and Second Opinions: Facilitate collaboration between radiologists to discuss potential discrepancies or errors and improve report accuracy.

# Challenges and Limitations:
1. Complexity of Medical Imaging: Ensuring the model's performance generalizes to new and diverse cases.
2. Ambiguity in Language: Capturing the subtleties in medical terminology for precise descriptions.
3. Integration with Existing Workflows: Seamlessly integrating the system into radiologists' existing practices.
4. Trust and Reliability: Gaining the trust of radiologists by providing accurate and reliable feedback consistently.
5. Privacy and Security: Adhering to strict privacy and security regulations for sensitive medical data.
6. Liability and Accountability: Establishing clear guidelines for accountability to address potential legal and ethical concerns.
7. Bias and Fairness: Mitigating potential biases in the model by training on diverse and representative data.

# Next Steps:
1. Develop a prototype of the image-text grounding system based on state-of-the-art techniques.
2. Evaluate the performance of the prototype on benchmark datasets and refine the model as needed.
3. Collaborate with radiologists to gather expert feedback and validate the system's effectiveness in real-world scenarios.
4. Address challenges and limitations iteratively, refining the model to ensure its practical applicability in the medical imaging domain.
5. Develop a plan for integrating the system with existing workflows and technologies in medical institutions.
6. By developing and refining this image-text grounding system, we aim to create a valuable tool for radiologists, enhancing their ability to generate accurate and meaningful descriptions of medical images while minimizing errors.

TODO:
* Get rid of the crossmodal attention --- DONE
* Visualize the loss evolution and accuracy during training, validation stages. --- DONE
* Debug and print all the dimensions and make sure the dimension sizes are the expected. --- DONE
* Run the experiments over again. --- DONE

* Train the VIT on the image classes, and use that vit instead of the vit_base_patch16_224 -- DONE
* Train on the much larger dataset that you are currently downloading. -- DONE


11th April Todo:
* thank God no one has published your work yet. (Remember deadline for ICLR is 17 April)
* while training is going on...
* read other papers and see how they reported their results. -- DONE 
* modify the introduction section of your paper, let it have a lot more recent stuff 
    * introduction
    * related work section 
    * methodology section 
    * think of some tables to add on the results that you (already) have
* figure out which other figures you might need to have in your paper 
* Have a list of 3 examples where you show that your model performs well. -- DONE
* Then a list of over 10 examples at the end of the paper showing all the examples that you have. -- Maybe 
* Read that twitter thread by Katie Link -- DONE (was not too impressed with current work)
* We shall submit the stuff on arxiv after the Friday Meeting with Carl -- PENDING
* Remember the number of image-text pairs you have is 377109 -- GOT
* Need to add description of the specific ViT architecture that you are using the timm library thing
* Need to talk about the batch size, learning rate, and device you use for training.
* Need to talk about 

11th April Todo: 
* Complete Discrete Math homework due tomorrow
* Start studying organic chemistry