//VERSION=3

function setup() {
    return {
        input: ["B02", "B03", "B04", "B11", "dataMask"],
        output: {bands: 1, sampleType: "FLOAT32"}
    };
}

function evaluatePixel(sample) {
    let NDSI = index(sample.B03, sample.B11);

    if ((NDSI > 0.38) && (sample.B04 > 0.18)) {
        return [100]
    } else {
        return [0]
    }
}
