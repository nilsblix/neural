import * as gui from "./gui/gui.ts";
import { Engine } from "./nnmanager.ts";
import { ImageDrawer } from "./imagedrawer.ts";
import { Network } from "./neural_network/nnetwork.ts";
import * as ml from "./mathlib/math.ts";

import { json1, json_728_64_32_10_88perc } from "./neural_network/examples.ts";

const enum UiAction {
	placeholder,
	begin_training,
	log_network,
	drag_brush_size,
	switch_draw_mode,
	render_image,
	increment_img_id,
	decrement_img_id,
	log_json_current_network,
	load_network_1,
	load_network_2,
	toggle_softmax,
}

const engine = new Engine(34);
const image_drawer = new ImageDrawer("f32", 28 * 28);

const c = <gui.REND>gui.canvas.getContext("2d");
const input_state = new gui.InputState(gui.canvas, 0, 0)

const nncanvas = <HTMLCanvasElement>document.getElementById("mnist-canvas");
const nnc = nncanvas.getContext("2d");

let nn_output = ml.Vector.fromType("f32", 10);

let img_id = 1;
let brush_size = 1;
let softmax = false;

function getSortedOutputArray() {
	const outputPairs = [];
	let index = 0;

	for (const value of nn_output.elements) {
		outputPairs.push({ digit: index, probability: value });
		index++;
	}

	outputPairs.sort((a, b) => b.probability - a.probability);
	return outputPairs;
}

const update = () => {

	if (nnc == undefined)
		return;


	gui.updateCanvasSizing();
	const stack = new gui.Stack<gui.N<UiAction>>();

	const wdraw = stack.makeWindow(c, input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { x: 800, title: "iteractivity options", width: 260, height: 210 });
	wdraw.makeLabel(c, null, "drawing options");
	wdraw.makeDraggable(c, UiAction.drag_brush_size, "brush size = " + brush_size);
	wdraw.makeButton(c, UiAction.switch_draw_mode, "mode: " + image_drawer.mode);

	wdraw.makeLabel(c, null, " ");
	wdraw.makeLabel(c, null, "render mnist images: ");
	wdraw.makeButton(c, UiAction.increment_img_id, "inc imagd id");
	wdraw.makeButton(c, UiAction.decrement_img_id, "dec imagd id");
	wdraw.makeButton(c, UiAction.render_image, "render transformed image, id=" + img_id);

	const wtraining = stack.makeWindow(c, input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { x: 800, title: "neural network options", y: 220, height: 500, width: 600 });

	wtraining.setMode("two columns", { min_width: 250, max_width: 400 });
	wtraining.makeButton(c, UiAction.toggle_softmax, "softmax = " + softmax);
	wtraining.makeLabel(c, null, "Results ==> ");
	wtraining.setMode("normal");

	const sorted_output = getSortedOutputArray();
	for (const { digit, probability } of sorted_output) {
		wtraining.makeLabel(c, null, digit + " --> " + (100 * probability).toFixed(2) + " %");
	}

	wtraining.makeLabel(c, null, " ");
	wtraining.makeLabel(c, null, "Current network options ==> ");
	wtraining.makeButton(c, UiAction.load_network_1, "load network 1 (random)");
	wtraining.makeButton(c, UiAction.load_network_2, "load network 2 88% (728 -> 64 -> 32 -> 10)");

	wtraining.makeLabel(c, null, " ");
	wtraining.makeLabel(c, null, "Training options ==> ");
	wtraining.makeButton(c, UiAction.log_json_current_network, "log current network as json");
	wtraining.makeButton(c, UiAction.begin_training, "begin training");
	wtraining.makeButton(c, UiAction.log_network, "log network");

	const ret = stack.requestAction(input_state);
	const action = ret.action;

	switch (action) {
		case UiAction.render_image:
			const img = engine.getImage(img_id);
			image_drawer.image_input = engine.transformImage(img.xs);
			nn_output = engine.evaluate(image_drawer.image_input);
			break;
		case UiAction.increment_img_id:
			img_id++;
			break;
		case UiAction.decrement_img_id:
			img_id--;
			break;
		case UiAction.begin_training:
			engine.train();
			break;
		case UiAction.log_network:
			console.dir(engine.network);
			break;
		case UiAction.switch_draw_mode:
			image_drawer.mode = image_drawer.mode == "draw" ? "erase" : "draw";
			break;
		case UiAction.drag_brush_size:
			brush_size = Number(gui.updateDraggableValue(brush_size, input_state, 0.02, { min: 0, max: 10 }).toFixed(3));
			break;
		case UiAction.log_json_current_network:
			console.log("'" + JSON.stringify(engine.network.toObject()) + "'");
			break;
		case UiAction.load_network_1:
			engine.network = Network.fromObject(JSON.parse(json1));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.load_network_2:
			engine.network = Network.fromObject(JSON.parse(json_728_64_32_10_88perc));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.toggle_softmax:
			softmax = !softmax;
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
	}

	if (JSON.stringify(input_state.active_widget_loc) == JSON.stringify([]) && input_state.mouse_down) {
		const rect = nnc.canvas.getBoundingClientRect();
		const x = input_state.mouse_position.x - rect.left;
		const y = input_state.mouse_position.y - rect.top;

		const x01 = x / rect.width;
		const y01 = y / rect.height;

		const x0c = x01 * nnc.canvas.width;
		const y0c = y01 * nnc.canvas.height;

		image_drawer.updateInput({ x: x0c, y: y0c }, 28, brush_size);
		nn_output = engine.evaluate(image_drawer.image_input);
	}

	engine.renderImage(image_drawer.image_input, nnc);

	c.clearRect(0, 0, c.canvas.width, c.canvas.height);
	stack.stack_render(c, input_state);

	input_state.end();

	requestAnimationFrame(update);
}

update();
