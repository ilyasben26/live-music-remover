use resvg::usvg::TreeParsing;

fn main() {
    let svg_data = std::fs::read("assets/logo.svg").expect("Failed to read assets/logo.svg");

    let options = resvg::usvg::Options::default();
    let usvg_tree =
        resvg::usvg::Tree::from_data(&svg_data, &options).expect("Failed to parse SVG");

    let size = 256u32;
    let svg_size = usvg_tree.size;
    let sx = size as f32 / svg_size.width();
    let sy = size as f32 / svg_size.height();
    let transform = resvg::tiny_skia::Transform::from_scale(sx, sy);

    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(size, size).expect("Failed to create pixmap");

    let tree = resvg::Tree::from_usvg(&usvg_tree);
    tree.render(transform, &mut pixmap.as_mut());

    pixmap
        .save_png("assets/logo.png")
        .expect("Failed to save logo.png");

    println!("cargo:rerun-if-changed=assets/logo.svg");
}
