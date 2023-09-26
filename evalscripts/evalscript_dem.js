//VERSION=3

function setup() {
    return {
        input: ["DEM", "dataMask"],
        output: {bands: 1, sampleType: "FLOAT32"}
    }
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return [NaN]
    } else {
        return [Math.round(sample.DEM)]
    }
}
