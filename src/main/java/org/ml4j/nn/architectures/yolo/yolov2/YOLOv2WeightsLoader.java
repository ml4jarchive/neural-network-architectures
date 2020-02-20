/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.architectures.yolo.yolov2;

import java.io.Serializable;

import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsVector;

/**
 * Interface for helper to load YOLO v2 weights
 * 
 * @author Michael Lavelle
 *
 */
public interface YOLOv2WeightsLoader extends Serializable {
	
	WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth);
	BiasVector getConvolutionalLayerBiases(String name, int outputDepth);
	WeightsVector getBatchNormLayerGamma(String name, int outputDepth);
	BiasVector getBatchNormLayerBeta(String name, int outputDepth);
	FeaturesVector getBatchNormLayerMovingVariance(String name, int outputDepth);
	FeaturesVector getBatchNormLayerMovingMean(String name, int outputDepth);


}
