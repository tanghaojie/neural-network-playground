/* 

==============================================================================*/

/**
 * 神经网络中的连接。
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string

  /**
   * 连接前节点
   */
  source: Node

  /**
   * 连接后节点
   */
  dest: Node

  /**
   * 连接权重（正向传播参与计算，反向传播后计算更新）。
   * 初始值一般随机，随着训练迭代一步一步优化。
   * 训练目的主要就是求这个值。
   */
  weight = Math.random() - 0.5

  /**
   * 指示是否死连接，即后续节点不再参与计算（反向传播）
   */
  isDead = false

  /**
   * 误差导数，调权重时使用。（反向传播）；
   * 计算规则：连接后节点的 inputDer * 连接前节点的 output。
   * Error derivative with respect to this weight. */
  errorDer = 0

  /**
   * 误差导数的累加值（反向传播）。就是 errorDer 求和。
   * 因为不一定训练每一步都更新节点参数，所以需要保存更新前的累加值。更新节点更新后重置。
   * Accumulated error derivative since the last update. */
  accErrorDer = 0

  /**
   * 误差导数的累加值计数（反向传播）。就是 accErrorDer 计算次数计数。
   * 因为不一定训练每一步都更新节点参数，所以需要保存更新前的累加值。更新节点更新后重置。
   * Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0

  /**
   * 正则化函数
   */
  regularization: RegularizationFunction

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(
    source: Node,
    dest: Node,
    regularization: RegularizationFunction,
    initZero?: boolean
  ) {
    this.id = source.id + '-' + dest.id
    this.source = source
    this.dest = dest
    this.regularization = regularization
    if (initZero) {
      this.weight = 0
    }
  }
}

/**
 * 神经网络中的节点。
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string

  /**
   * 输入连接。
   * List of input links. */
  inputLinks: Link[] = []

  /**
   * 偏置参数，还看不懂。
   */
  bias = 0.1

  /**
   * 输出连接。
   * List of output links. */
  outputs: Link[] = []

  /**
   * 输入总和，线性累加（正向传播）。
   * w1x1 + w2x2 + ... +wnxn + b；
   * 即，各个 前节点的输出 * 连接权重 的总和。
   */
  totalInput: number

  /**
   * 节点输出（正向传播）。
   * 对于输入层，输出就是输入层函数计算结果。
   * 对于其他层，输出通过下面 updateOutput 方法计算，
   * 实际就是 activation(totalInput)。
   */
  output: number

  /**
   * 节点输出误差导数值（反向传播）。
   * 对于输出层（最后一层），使用误差导数（误差函数的导函数）计算输出值和目标值差异 errorFunc.der(outputNode.output, target)。
   * 对于其他层，计算当前节点  outputLink.weight * outputLink.dest.inputDer。
   * Error derivative with respect to this node's output. */
  outputDer = 0

  /**
   * 输入总和的误差导数值（反向传播）。 outputDer * activation.der(totalInput)。
   * Error derivative with respect to this node's total input. */
  inputDer = 0

  /**
   * 误差导数的累加值（反向传播）。就是 inputDer 求和。
   * 因为不一定训练每一步都更新节点参数，所以需要保存更新前的累加值。更新节点更新后重置。
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0

  /**
   * 误差导数的累加值计数（反向传播）。就是 inputDer 计算次数计数。
   * 因为不一定训练每一步都更新节点参数，所以需要保存更新前的累加值。更新节点更新后重置。
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0

  /**
   * 激活函数
   *  Activation function that takes total input and returns node's output */
  activation: ActivationFunction

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id
    this.activation = activation
    if (initZero) {
      this.bias = 0
    }
  }

  /**
   * 计算输出。（正向传播）
   * Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j]
      this.totalInput += link.weight * link.source.output
    }
    this.output = this.activation.output(this.totalInput)
    return this.output
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number
  der: (output: number, target: number) => number
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number
  der: (input: number) => number
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number
  der: (weight: number) => number
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    // 均方差
    error: (output: number, target: number) => 0.5 * Math.pow(output - target, 2),
    // 差值越大，梯度越大
    der: (output: number, target: number) => output - target,
  }
}

/** Polyfill for TANH */
;(Math as any).tanh =
  (Math as any).tanh ||
  function (x) {
    if (x === Infinity) {
      return 1
    } else if (x === -Infinity) {
      return -1
    } else {
      let e2x = Math.exp(2 * x)
      return (e2x - 1) / (e2x + 1)
    }
  }

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: (x) => (Math as any).tanh(x),
    der: (x) => {
      let output = Activations.TANH.output(x)
      return 1 - output * output
    },
  }
  public static RELU: ActivationFunction = {
    output: (x) => Math.max(0, x),
    der: (x) => (x <= 0 ? 0 : 1),
  }
  public static SIGMOID: ActivationFunction = {
    output: (x) => 1 / (1 + Math.exp(-x)),
    der: (x) => {
      let output = Activations.SIGMOID.output(x)
      return output * (1 - output)
    },
  }
  public static LINEAR: ActivationFunction = {
    output: (x) => x,
    der: (x) => 1,
  }
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: (w) => Math.abs(w),
    der: (w) => (w < 0 ? -1 : w > 0 ? 1 : 0),
  }
  public static L2: RegularizationFunction = {
    output: (w) => 0.5 * w * w,
    der: (w) => w,
  }
}

/**
 *
 * 创建神经网络，配置一堆超参数。
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
  networkShape: number[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
  regularization: RegularizationFunction,
  inputIds: string[],
  initZero?: boolean
): Node[][] {
  let numLayers = networkShape.length
  let id = 1
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = []
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1
    let isInputLayer = layerIdx === 0
    let currentLayer: Node[] = []
    network.push(currentLayer)
    let numNodes = networkShape[layerIdx]
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString()
      if (isInputLayer) {
        nodeId = inputIds[i]
      } else {
        id++
      }
      let node = new Node(nodeId, isOutputLayer ? outputActivation : activation, initZero)
      currentLayer.push(node)
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j]
          let link = new Link(prevNode, node, regularization, initZero)
          prevNode.outputs.push(link)
          node.inputLinks.push(link)
        }
      }
    }
  }
  return network
}

/**
 * 正向传播
 *
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network 节点网络：输入层+隐藏层+输出层。例如：[3, 4, 5, 1]，代表：输入层有3个节点；隐藏层有两层，一层4个节点，一层5个节点；输出层 1 层是固定值。
 * @param inputs 输入层各个节点的计算值。The input array. Its length should match the number of input
 *     nodes in the network.
 * @return 整个网络的输出结果。The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0]
  if (inputs.length !== inputLayer.length) {
    throw new Error('The number of inputs must match the number of nodes in' + ' the input layer')
  }
  // 把 inputs 写入输入层各个节点的output。Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i]
    node.output = inputs[i]
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx]
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i]
      node.updateOutput()
    }
  }
  return network[network.length - 1][0].output
}

/**
 * 反向传播
 * 使用提供的 target（应该的正确结果）和各个节点的 output， 运行反向传播。
 * 反向传播修改神经网路内部的状态参数值：每个节点的 误差导数（误差函数的导数）、每个节点的连接权重。
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 *
 * @param network 和正向传播的一样
 * @param target 训练应该得到的实际值
 * @param errorFunc 误差函数
 */
export function backProp(network: Node[][], target: number, errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  // 输出层的 outputDer 单独计算
  let outputNode = network[network.length - 1][0]
  outputNode.outputDer = errorFunc.der(outputNode.output, target)

  // Go through the layers backwards.
  // 从输出层开始，从后往前依次计算
  // 输入层（第一层）不参与计算
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx]
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    // 当前层 通过激活函数的导函数计算 输入误差
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i]
      node.inputDer = node.outputDer * node.activation.der(node.totalInput)
      node.accInputDer += node.inputDer
      node.numAccumulatedDers++
    }

    // Error derivative with respect to each weight coming into the node.
    // 当前层 前序连接 参数计算并保存，为后续调参计算 w 做准备
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i]
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j]
        // 死链接不参与计算
        if (link.isDead) {
          continue
        }
        link.errorDer = node.inputDer * link.source.output
        link.accErrorDer += link.errorDer
        link.numAccumulatedDers++
      }
    }
    if (layerIdx === 1) {
      continue
    }
    // 当前层 前序层 outputDer 更新
    let prevLayer = network[layerIdx - 1]
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i]
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0
      for (let j = 0; j < node.outputs.length; j++) {
        let outputLink = node.outputs[j]
        node.outputDer += outputLink.weight * outputLink.dest.inputDer
      }
    }
  }
}

/**
 * 参数更新函数。
 * 一般训练 N 次以后，才更新一次参数
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number, regularizationRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx]
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i]
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        // 训练 N 次后的平均值
        node.bias -= (learningRate * node.accInputDer) / node.numAccumulatedDers
        node.accInputDer = 0
        node.numAccumulatedDers = 0
      }
      // Update the weights coming into this node.
      // 更新所有连接的 w，并且重置一些参数
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j]
        // 死节点不再参与
        if (link.isDead) {
          continue
        }
        // 如果设定了正则化，计算正则化
        let regulDer = link.regularization ? link.regularization.der(link.weight) : 0
        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          // 更新 w。
          // 实际上就是累计误差accErrorDer * 学习率，求平均值
          link.weight = link.weight - (learningRate / link.numAccumulatedDers) * link.accErrorDer
          // Further update the weight based on regularization.
          // 基于正则化的结果，对于过拟合的调整，拉回来一些
          let newLinkWeight = link.weight - learningRate * regularizationRate * regulDer
          if (
            link.regularization === RegularizationFunction.L1 &&
            link.weight * newLinkWeight < 0
          ) {
            // 对于这个条件，节点过拟合了，是无意义的，所以置为死节点。
            // weight 和 newLinkWeight 一个正数一个负数，说明 weight 已经很接近 0 了。
            // 因为在更新模型参数时，算法会计算梯度，并按照梯度的方向进行调整，直到接近 0 。
            // 这意味着这些特征对于模型的预测没有任何贡献。
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0
            link.isDead = true
          } else {
            link.weight = newLinkWeight
          }
          // 重置计数器
          link.accErrorDer = 0
          link.numAccumulatedDers = 0
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(
  network: Node[][],
  ignoreInputs: boolean,
  accessor: (node: Node) => any
) {
  for (let layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx]
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i]
      accessor(node)
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0]
}
