import { XYZPointTypes, PointXYZ } from '@/modules/common/point-types';
import { PointCloud, PointIndices, Indices } from '@/modules/common/PointCloud';
import ModelCoefficients from '@/modules/common/ModelCoefficients';
import { UnionToIntersection } from '@/types';
import PCLBase from '@/modules/common/PCLBase';
import { SacMethodTypes, SacModelTypes } from '@/modules/sample-consensus/constants';
import Search from '@/modules/search/Search';
import { toXYZPointCloud } from '@/modules/common/utils';

class SACSegmentation<
  T extends XYZPointTypes = PointXYZ & Partial<UnionToIntersection<XYZPointTypes>>,
> extends PCLBase<T> {
  private _modelType: SacModelTypes = 0;
  private _methodType: SacMethodTypes = 0;
  private _optimizeCoefficients = true;
  private _epsAngle = 0;
  private _threshold = 0;
  private _maxIterations = 50;
  private _probability = 0.99;
  private _samplesRadius = 0;
  private _samplesRadiusSearch?: Search<T>;
  private _minRadius = Number.MIN_VALUE;
  private _maxRadius = Number.MAX_VALUE;

  constructor(random = false) {
    const native = new __PCLCore__.SACSegmentationPointXYZ(random);
    super(native);
  }

  public setModelType(model: SacModelTypes) {
    this._native.setModelType(model);
    this._modelType = model;
  }

  public getModelType() {
    return this._modelType;
  }

  public setMethodType(method: SacMethodTypes) {
    this._native.setMethodType(method);
    this._methodType = method;
  }

  public getMethodType() {
    return this._methodType;
  }

  public setDistanceThreshold(threshold: number) {
    this._native.setDistanceThreshold(threshold);
    this._threshold = threshold;
  }

  public getDistanceThreshold() {
    return this._threshold;
  }

  public setMaxIterations(maxIterations: number) {
    this._native.setMaxIterations(maxIterations);
    this._maxIterations = maxIterations;
  }

  public getMaxIterations() {
    return this._maxIterations;
  }

  public setProbability(probability: number) {
    this._native.setProbability(probability);
    this._probability = probability;
  }

  public getProbability() {
    return this._probability;
  }

  public setRadiusLimits(minRadius: number, maxRadius: number) {
    this._native.setRadiusLimits(minRadius, maxRadius);
    this._minRadius = minRadius;
    this._maxRadius = maxRadius;
  }

  public getRadiusLimits() {
    return {
      minRadius: this._minRadius,
      maxRadius: this._maxRadius,
    };
  }

  public setSamplesMaxDist(radius: number, search: Search<T>) {
    this._native.setSamplesMaxDist(radius, search._native);
    this._samplesRadius = radius;
    this._samplesRadiusSearch = search;
  }

  public getSamplesMaxDist() {
    return {
      radius: this._samplesRadius,
      search: this._samplesRadiusSearch,
    };
  }

  public setEpsAngle(epsAngle: number) {
    this._native.setEpsAngle(epsAngle);
    this._epsAngle = epsAngle;
  }

  public getEpsAngle() {
    return this._epsAngle;
  }

  public setOptimizeCoefficients(optimizeCoefficients: boolean) {
    this._native.setOptimizeCoefficients(optimizeCoefficients);
    this._optimizeCoefficients = optimizeCoefficients;
  }

  public getOptimizeCoefficients() {
    return this._optimizeCoefficients;
  }

  public segment(inliers: PointIndices, modelCoefficients: ModelCoefficients) {
    this._native.segment(inliers._native, modelCoefficients._native);

    // 确保内点索引被正确地同步回JavaScript对象
    // 由于WebAssembly绑定可能没有正确地更新inliers.indices，我们需要手动同步
    const nativeIndices = inliers._native.indices;
    if (nativeIndices) {
      // 检查一下内点的数量
      const indicesSize = nativeIndices.size();
      console.debug(`Segmentation found ${indicesSize} inliers`);

      // 如果有内点，我们需要更新indices属性
      if (indicesSize > 0) {
        // 创建一个新的Indices对象
        const newIndices = new Indices();

        // 从原生对象中复制数据
        for (let i = 0; i < indicesSize; i++) {
          newIndices.push(nativeIndices.get(i));
        }

        // 更新inliers的indices属性
        inliers.indices = newIndices;
      }
    }
  }

  /**
   * Only supports `PointXYZ` type,
   * if it is not `PointXYZ` type, it will use `toXYZPointCloud` method to convert to `PointXYZ`
   */
  public setInputCloud(cloud: PointCloud<T>) {
    if (cloud._PT === PointXYZ) {
      this._native.setInputCloud(cloud._native);
    } else {
      const cloudXYZ = new PointCloud<PointXYZ>();
      toXYZPointCloud(cloud, cloudXYZ);
      this._native.setInputCloud(cloudXYZ._native);
      cloudXYZ.manager.delete();
    }
  }
}

export default SACSegmentation;
