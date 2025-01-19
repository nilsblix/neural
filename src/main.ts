import * as gui from "./gui/gui.ts";
import { Engine } from "./nnmanager.ts";

const enum UiAction {
	placeholder,
	increment,
	decrement,
	drag_seed,
	begin_training,
	log_network,
	drag_image_id,
	render_image,
}

const engine = new Engine(16);

const c = <gui.REND>gui.canvas.getContext("2d");
const input_state = new gui.InputState(gui.canvas, 0, 0)

const nncanvas = <HTMLCanvasElement>document.createElement("canvas");
nncanvas.id = "nncanvas";
const nnc = nncanvas.getContext("2d");

nncanvas.style.top = 0 + "";
nncanvas.style.left = 0 + "";
nncanvas.style.position = "absolute";

let num = 0;
let seed = 1;
let image_id = 1;

const update = () => {

	gui.updateCanvasSizing();
	const stack = new gui.Stack<gui.N<UiAction>>();

	const w = stack.makeWindow(c, input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { title: "neural test", width: 300, height: 400 });

	w.makeLabel(c, null, "num = " + num);
	w.makeButton(c, UiAction.increment, "increment");
	w.makeButton(c, UiAction.decrement, "decrement");

	w.makeLabel(c, null, "seed = " + seed);
	w.makeDraggable(c, UiAction.drag_seed, "drag seed");

	w.makeLabel(c, null, " ");
	w.makeButton(c, UiAction.begin_training, "begin training");
	w.makeButton(c, UiAction.log_network, "log network");

	w.makeDraggable(c, UiAction.drag_image_id, "imageid = " + image_id);
	w.makeButton(c, UiAction.render_image, "render image");

	const ret = stack.requestAction(input_state);
	const action = ret.action;


	switch (action) {
		case UiAction.increment:
			num++;
			break;
		case UiAction.decrement:
			num--;
			break;
		case UiAction.drag_seed:
			seed = gui.updateDraggableValue(seed, input_state, 1.0);
			break;
		case UiAction.begin_training:
			engine.train();
			break;
		case UiAction.log_network:
			console.dir(engine.network);
			break;
		case UiAction.drag_image_id:
			image_id = gui.updateDraggableValue(image_id, input_state, 1.0);
			break;
		case UiAction.render_image:
			if (nnc != undefined)
				engine.renderImage(image_id, nnc);
			break;

	}

	c.clearRect(0, 0, c.canvas.width, c.canvas.height);
	stack.stack_render(c, input_state);

	input_state.end();

	requestAnimationFrame(update);
}

update();
