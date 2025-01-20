import * as tf from '@tensorflow/tfjs';

export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;

const MNIST_IMAGES_SPRITE_PATH =
	'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
	'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */
export class MnistData {
	trainImages: Float32Array;
	datasetImages: Float32Array;
	testImages: Float32Array;
	trainLabels: Uint8Array;
	datasetLabels: Uint8Array;
	testLabels: Uint8Array;

	constructor() {
		this.trainImages = new Float32Array();
		this.datasetImages = new Float32Array();
		this.testImages = new Float32Array();
		this.trainLabels = new Uint8Array();
		this.datasetLabels = new Uint8Array();
		this.testLabels = new Uint8Array();
	}

	async load() {
		// Make a request for the MNIST sprited image.
		const img = new Image();
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d', { willReadFrequently: true });

		if (ctx == undefined) {
			throw new Error("mnist ctx did not load properly");
		}

		const imgRequest = new Promise<void>((resolve) => {
			img.crossOrigin = '';
			img.onload = () => {
				img.width = img.naturalWidth;
				img.height = img.naturalHeight;

				const datasetBytesBuffer =
					new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

				const chunkSize = 5000;
				canvas.width = img.width;
				canvas.height = chunkSize;

				for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
					const datasetBytesView = new Float32Array(
						datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
						IMAGE_SIZE * chunkSize);
					ctx.drawImage(
						img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
						chunkSize);

					const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

					for (let j = 0; j < imageData.data.length / 4; j++) {
						// All channels hold an equal value since the image is grayscale, so
						// just read the red channel.
						datasetBytesView[j] = imageData.data[j * 4] / 255;
					}
				}
				this.datasetImages = new Float32Array(datasetBytesBuffer);

				resolve();
			};
			img.src = MNIST_IMAGES_SPRITE_PATH;
		});

		const labelsRequest = fetch(MNIST_LABELS_PATH).then((response) => response.arrayBuffer());
		const [imgRequestResult, labelsResponseArrayBuffer] =
			await Promise.all([imgRequest, labelsRequest]);

		imgRequestResult;

		this.datasetLabels = new Uint8Array(labelsResponseArrayBuffer);

		// Slice the the images and labels into train and test sets.
		this.trainImages =
			this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
		this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
		this.trainLabels =
			this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
		this.testLabels =
			this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
	}

	/**
	 * Get all training data as a data tensor and a labels tensor.
	 *
	 * @returns
	 *   xs: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
	 *   labels: The one-hot encoded labels tensor, of shape
	 *     `[numTrainExamples, 10]`.
	 */
	getTrainData() {
		const xs = tf.tensor4d(
			this.trainImages,
			[this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
		const labels = tf.tensor2d(
			this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);
		return { xs, labels };
	}

	/**
	 * Get all test data as a data tensor and a labels tensor.
	 *
	 * @param {number} numExamples Optional number of examples to get. If not
	 *     provided,
	 *   all test examples will be returned.
	 * @returns
	 *   xs: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
	 *   labels: The one-hot encoded labels tensor, of shape
	 *     `[numTestExamples, 10]`.
	 */
	getTestData(numExamples: number) {
		let xs = tf.tensor4d(
			this.testImages,
			[this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
		let labels = tf.tensor2d(
			this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);

		if (numExamples != null) {
			xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
			labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
		}
		return { xs, labels };
	}
}
