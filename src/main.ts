//import * as gui from "./gui/gui.ts";
import * as gui from "./nvb-imgui/src/gui/gui.ts";
import { Engine } from "./nnmanager.ts";
import { ImageDrawer } from "./imagedrawer.ts";
import { Network } from "./neural_network/nnetwork.ts";
import * as ml from "./mathlib/math.ts";

import * as demos from "./neural_network/examples.ts";

const enum UiAction {
	placeholder,
	begin_training,
	log_network,
	drag_brush_size,
	drag_draw_rate,
	switch_draw_mode,
	render_image,
	increment_img_id,
	decrement_img_id,
	log_json_current_network,
	load_network_1,
	load_network_2,
	load_network_3,
	load_network_4,
	toggle_softmax,

}

const engine = new Engine(34);
const image_drawer = new ImageDrawer("f32", 28 * 28);

const nncanvas = <HTMLCanvasElement>document.getElementById("mnist-canvas");
const nnc = nncanvas.getContext("2d");

let nn_output = ml.Vector.fromType("f32", 10);

let img_id = 1;
let brush_size = 2.0;
let draw_rate = 0.1;
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

	const wdraw = stack.makeWindow(gui.c, gui.input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { x: 20, y: 40, title: "interactivity options", width: 280, height: 250 });
	wdraw.makeLabel(gui.c, null, "drawing options");
	wdraw.makeDraggable(gui.c, UiAction.drag_brush_size, "brush size = " + brush_size);
	wdraw.makeDraggable(gui.c, UiAction.drag_draw_rate, "draw rate = " + draw_rate);
	wdraw.makeButton(gui.c, UiAction.switch_draw_mode, "mode: " + image_drawer.mode);

	wdraw.makeLabel(gui.c, null, " ");
	wdraw.makeLabel(gui.c, null, "render mnist images: ");
	wdraw.makeButton(gui.c, UiAction.increment_img_id, "inc imagd id");
	wdraw.makeButton(gui.c, UiAction.decrement_img_id, "dec imagd id");
	wdraw.makeButton(gui.c, UiAction.render_image, "render transformed image, id=" + img_id);

	const wtraining = stack.makeWindow(gui.c, gui.input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { x: 10, title: "neural network options", y: 320, height: 465, width: 495 });

	wtraining.makeButton(gui.c, UiAction.toggle_softmax, "softmax = " + softmax);
	wtraining.makeLabel(gui.c, null, "Results ==> ");

	const sorted_output = getSortedOutputArray();
	for (const { digit, probability } of sorted_output) {
		wtraining.makeLabel(gui.c, null, digit + " --> " + (100 * probability).toFixed(2) + " %");
	}

	wtraining.makeLabel(gui.c, null, " ");
	wtraining.makeLabel(gui.c, null, "Current network options:");
	const width = 500;
	const half = width / 2;
	wtraining.setMode("two columns", { min_width: width, max_width: width });

	const y_offset = 3.0
	wtraining.makeText(gui.c, null, "Load a randomly generated network. I do not know the network's size", half);
	wtraining.cursor.y -= y_offset;
	wtraining.makeButton(gui.c, UiAction.load_network_1, "Unknown size");

	wtraining.makeText(gui.c, null, "Load 88% performing network", half);
	wtraining.cursor.y -= y_offset;
	wtraining.makeButton(gui.c, UiAction.load_network_2, "Size: 784 -> 64 -> 32 -> 10");

	wtraining.makeText(gui.c, null, "Load 81% performing network", half);
	wtraining.cursor.y -= y_offset;
	wtraining.makeButton(gui.c, UiAction.load_network_3, "Size: 784 -> 48 -> 16 -> 10");

	wtraining.makeText(gui.c, null, "Load 87% performing network", half);
	wtraining.cursor.y -= y_offset;
	wtraining.makeButton(gui.c, UiAction.load_network_4, "Size: 784 -> 48 -> 16 -> 10");

	wtraining.setMode("normal");

	wtraining.makeLabel(gui.c, null, " ");
	wtraining.makeLabel(gui.c, null, "Training options:");
	wtraining.makeButton(gui.c, UiAction.log_json_current_network, "log current network as json");
	wtraining.makeButton(gui.c, UiAction.begin_training, "begin training");
	wtraining.makeButton(gui.c, UiAction.log_network, "log network");

	const ret = stack.requestAction(gui.input_state);
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
			brush_size = Number(gui.updateDraggableValue(brush_size, gui.input_state, 0.02, { min: 0, max: 10 }).toFixed(3));
			break;
		case UiAction.drag_draw_rate:
			draw_rate = Number(gui.updateDraggableValue(draw_rate, gui.input_state, 0.005, { min: 0, max: 1 }));
			break;
		case UiAction.log_json_current_network:
			console.log("'" + JSON.stringify(engine.network.toObject()) + "'");
			break;
		case UiAction.load_network_1:
			engine.network = Network.fromObject(JSON.parse(demos.json1));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.load_network_2:
			engine.network = Network.fromObject(JSON.parse(demos.json_728_64_32_10_88perc));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.load_network_3:
			engine.network = Network.fromObject(JSON.parse(demos.json_basic_784_48_16_10));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.load_network_4:
			engine.network = Network.fromObject(JSON.parse(demos.json_784_48_16_10_87perc));
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
		case UiAction.toggle_softmax:
			softmax = !softmax;
			nn_output = engine.evaluate(image_drawer.image_input, softmax);
			break;
	}

	if (JSON.stringify(gui.input_state.active_widget_loc) == JSON.stringify([]) && gui.input_state.mouse_down) {
		const rect = nnc.canvas.getBoundingClientRect();
		const x = gui.input_state.mouse_position.x - rect.left;
		const y = gui.input_state.mouse_position.y - rect.top;

		const x01 = x / rect.width;
		const y01 = y / rect.height;

		const x0c = x01 * nnc.canvas.width;
		const y0c = y01 * nnc.canvas.height;

		image_drawer.updateInput({ x: x0c, y: y0c }, 28, brush_size, draw_rate);
		nn_output = engine.evaluate(image_drawer.image_input);
	}

	engine.renderImage(image_drawer.image_input, nnc);

	gui.c.clearRect(0, 0, gui.c.canvas.width, gui.c.canvas.height);
	stack.stack_render(gui.c, gui.input_state);

	gui.input_state.end();

	requestAnimationFrame(update);
}

update();
