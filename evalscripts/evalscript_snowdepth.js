//VERSION=3

function setup() {
    return {
        input: ["HS", "dataMask"],
        output: {bands: 1, sampleType: "FLOAT32"},
    }
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return [NaN]
    } else {
        return [sample.HS]
    }
}
