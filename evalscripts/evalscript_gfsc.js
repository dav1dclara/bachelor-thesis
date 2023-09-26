//VERSION=3

function setup() {
    return {
        input: ["GF", "dataMask"],
        output: {bands: 1, sampleType: "FLOAT32"},
    }
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0 || sample.GF > 100) {
        return [NaN]
    } else {
        return [sample.GF]
    }
}
