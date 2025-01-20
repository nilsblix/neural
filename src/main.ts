import * as gui from "./gui/gui.ts";
import { Engine } from "./nnmanager.ts";
import { ImageDrawer } from "./imagedrawer.ts";
import { Network } from "./neural_network/nnetwork.ts";
import * as ml from "./mathlib/math.ts";

import { json1 } from "./neural_network/examples.ts";

const enum UiAction {
	placeholder,
	begin_training,
	log_network,
	switch_draw_mode,
	render_image,
	increment_img_id,
	decrement_img_id,
	log_json_current_network,
	load_network_1,
}

//const vec = new ml.Vector(new Float32Array([1, 2, 3]));
//
//const json_vec = vec.toJSON();
//const parsed_vec = ml.Vector.fromJSON(json_vec);
//
//console.log("orig:", vec);
//console.log("json:", json_vec);
//console.log("parsed:", parsed_vec);
//
//const mat = new ml.Matrix([
//	new ml.Vector(new Float32Array([1, 2, 3])),
//	new ml.Vector(new Float32Array([4, 5, 6])),
//	new ml.Vector(new Float32Array([7, 8, 9])),
//]);
//
//const json_mat = mat.toJSON();
//const parsed_mat = ml.Matrix.fromJSON(json_mat);
//
//console.log("");
//
//console.log("mat orig:", mat);
//console.log("mat json:", json_mat);
//console.log("mat parsed:", parsed_mat);
//
//console.log("");
//
//console.log("ORIG mat * vec:", (ml.Matrix.multVector(mat, vec)));
//console.log("PARSED mat * vec:", (ml.Matrix.multVector(parsed_mat, parsed_vec)));

const engine = new Engine(34);
const image_drawer = new ImageDrawer("f32", 28 * 28);

const c = <gui.REND>gui.canvas.getContext("2d");
const input_state = new gui.InputState(gui.canvas, 0, 0)

const nncanvas = <HTMLCanvasElement>document.getElementById("mnist-canvas");
const nnc = nncanvas.getContext("2d");

let nn_output = ml.Vector.fromType("f32", 10);
let highest_activation = -1;

let img_id = 1;

const update = () => {

	if (nnc == undefined)
		return;


	gui.updateCanvasSizing();
	const stack = new gui.Stack<gui.N<UiAction>>();

	const w = stack.makeWindow(c, input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { x: 800, title: "neural test", width: 300, height: 600 });

	w.makeButton(c, UiAction.increment_img_id, "inc imagd id");
	w.makeButton(c, UiAction.decrement_img_id, "inc imagd id");
	w.makeButton(c, UiAction.render_image, "render transformed image, id=" + img_id);

	w.makeLabel(c, null, "");
	w.makeButton(c, UiAction.log_json_current_network, "log current network as json");
	w.makeButton(c, UiAction.load_network_1, "load network 1 (random)");

	w.makeLabel(c, null, " ");
	w.makeButton(c, UiAction.begin_training, "begin training");
	w.makeButton(c, UiAction.log_network, "log network");

	w.makeLabel(c, null, "");
	w.makeButton(c, UiAction.switch_draw_mode, "mode: " + image_drawer.mode);

	for (let i = 0; i < nn_output.length; i++) {
		w.makeLabel(c, null, (highest_activation == i ? "***" : "") + "% of " + i + " = " + nn_output.elements[i]);
	}

	const ret = stack.requestAction(input_state);
	const action = ret.action;

	switch (action) {
		case UiAction.render_image:
			const img = engine.getImage(img_id);
			image_drawer.image_input = engine.transformImage(img.xs);
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
		case UiAction.log_json_current_network:
			console.log("'" + JSON.stringify(engine.network.toObject()) + "'");
			break;
		case UiAction.load_network_1:
			engine.network = Network.fromObject(JSON.parse(json1));
			break;
	}

	if (JSON.stringify(input_state.active_widget_loc) == JSON.stringify([]) && input_state.mouse_down) {
		const rect = nnc.canvas.getBoundingClientRect();
		const x = input_state.mouse_position.x - rect.left;
		const y = input_state.mouse_position.y - rect.top;
		image_drawer.updateInput({ x: y, y: x }, 28);
		nn_output = engine.evaluate(image_drawer.image_input);

		var temp = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < nn_output.length; i++) {
			if (temp < nn_output.elements[i]) {
				temp = nn_output.elements[i];
				highest_activation = i;
			}
		}
	}

	engine.renderImage(image_drawer.image_input, nnc);

	c.clearRect(0, 0, c.canvas.width, c.canvas.height);
	stack.stack_render(c, input_state);

	input_state.end();

	requestAnimationFrame(update);
}

update();
