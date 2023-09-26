//VERSION=3

function setup() {
    return {
        input: ["HS", "dataMask"],
        output: {bands: 1, sampleType: "FLOAT32"},
        mosaicking: "ORBIT"
    }
}

function isClear(sample) {
    return sample.dataMask === 1;
}

function sum(array) {
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
        sum += array[i].HS;
    }
    return sum
}

function evaluatePixel(samples) {
    const clearTs = samples.filter(isClear)
    const mean = sum(clearTs) / clearTs.length
    return [mean]
}