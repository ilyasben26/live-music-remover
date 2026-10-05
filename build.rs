use resvg::usvg::TreeParsing;

/// Sizes packed into the executable's icon, so Windows can pick a sharp one
/// for each place it shows it (taskbar, Explorer views, title bar).
const ICO_SIZES: [u32; 7] = [16, 24, 32, 48, 64, 128, 256];

fn render(tree: &resvg::Tree, svg_size: resvg::usvg::Size, size: u32) -> resvg::tiny_skia::Pixmap {
    let sx = size as f32 / svg_size.width();
    let sy = size as f32 / svg_size.height();
    let transform = resvg::tiny_skia::Transform::from_scale(sx, sy);
    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(size, size).expect("Failed to create pixmap");
    tree.render(transform, &mut pixmap.as_mut());
    pixmap
}

/// Builds an .ico file holding one PNG-encoded image per size.
fn build_ico(tree: &resvg::Tree, svg_size: resvg::usvg::Size) -> Vec<u8> {
    let pngs: Vec<(u32, Vec<u8>)> = ICO_SIZES
        .iter()
        .map(|&s| {
            let png = render(tree, svg_size, s)
                .encode_png()
                .expect("Failed to encode icon PNG");
            (s, png)
        })
        .collect();

    let mut ico = Vec::new();
    // ICONDIR: reserved, type (1 = icon), image count.
    ico.extend_from_slice(&0u16.to_le_bytes());
    ico.extend_from_slice(&1u16.to_le_bytes());
    ico.extend_from_slice(&(pngs.len() as u16).to_le_bytes());

    let mut offset = 6 + 16 * pngs.len() as u32;
    for (size, png) in &pngs {
        // ICONDIRENTRY: a width/height of 0 means 256.
        let dim = if *size >= 256 { 0 } else { *size as u8 };
        ico.push(dim);
        ico.push(dim);
        ico.push(0); // palette size
        ico.push(0); // reserved
        ico.extend_from_slice(&1u16.to_le_bytes()); // color planes
        ico.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
        ico.extend_from_slice(&offset.to_le_bytes());
        offset += png.len() as u32;
    }
    for (_, png) in &pngs {
        ico.extend_from_slice(png);
    }
    ico
}

fn main() {
    let svg_data = std::fs::read("assets/logo.svg").expect("Failed to read assets/logo.svg");

    let options = resvg::usvg::Options::default();
    let usvg_tree =
        resvg::usvg::Tree::from_data(&svg_data, &options).expect("Failed to parse SVG");
    let svg_size = usvg_tree.size;
    let tree = resvg::Tree::from_usvg(&usvg_tree);

    // Window icon, loaded at runtime by `load_icon`.
    render(&tree, svg_size, 256)
        .save_png("assets/logo.png")
        .expect("Failed to save logo.png");

    // Executable icon, shown in Explorer and on shortcuts.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let ico_path = out_dir.join("logo.ico");
        std::fs::write(&ico_path, build_ico(&tree, svg_size)).expect("Failed to write logo.ico");
        winresource::WindowsResource::new()
            .set_icon(ico_path.to_str().unwrap())
            .compile()
            .expect("Failed to embed the executable icon");
    }

    println!("cargo:rerun-if-changed=assets/logo.svg");
}
