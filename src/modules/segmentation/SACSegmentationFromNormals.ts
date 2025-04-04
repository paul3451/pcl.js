import { XYZPointTypes, PointXYZ, Normal } from '@/modules/common/point-types';
import { PointCloud, PointIndices, Indices } from '@/modules/common/PointCloud';
import ModelCoefficients from '@/modules/common/ModelCoefficients';
import { UnionToIntersection } from '@/types';
import SACSegmentation from './SACSegmentation';
import { toXYZPointCloud } from '@/modules/common/utils';

class SACSegmentationFromNormals<
  T extends XYZPointTypes = PointXYZ & Partial<UnionToIntersection<XYZPointTypes>>,
> extends SACSegmentation<T> {
  private _normalDistanceWeight = 0.1;
  private _distanceFromOrigin = 0;
  private _minAngle = 0;
  private _maxAngle = Math.PI / 2;

  constructor(random = false) {
    const native = new __PCLCore__.SACSegmentationFromNormalsPointXYZNormal(random);
    super(random);
    this._native = native;
  }

  public setInputNormals(normals: PointCloud<Normal>) {
    this._native.setInputNormals(normals._native);
  }

  public getInputNormals(): PointCloud<Normal> {
    const native = this._native.getInputNormals();
    const cloud = new PointCloud<Normal>();
    cloud._native = native;
    return cloud;
  }

  public setNormalDistanceWeight(weight: number) {
    this._native.setNormalDistanceWeight(weight);
    this._normalDistanceWeight = weight;
  }

  public getNormalDistanceWeight() {
    return this._normalDistanceWeight;
  }

  public setMinMaxOpeningAngle(minAngle: number, maxAngle: number) {
    this._native.setMinMaxOpeningAngle(minAngle, maxAngle);
    this._minAngle = minAngle;
    this._maxAngle = maxAngle;
  }

  public getMinMaxOpeningAngle() {
    return {
      minAngle: this._minAngle,
      maxAngle: this._maxAngle,
    };
  }

  public setDistanceFromOrigin(distance: number) {
    this._native.setDistanceFromOrigin(distance);
    this._distanceFromOrigin = distance;
  }

  public getDistanceFromOrigin() {
    return this._distanceFromOrigin;
  }

  /**
   * 覆盖segment方法以确保内点索引被正确同步
   */
  public segment(inliers: PointIndices, modelCoefficients: ModelCoefficients) {
    this._native.segment(inliers._native, modelCoefficients._native);

    // 确保内点索引被正确地同步回JavaScript对象
    // 由于WebAssembly绑定可能没有正确地更新inliers.indices，我们需要手动同步
    const nativeIndices = inliers._native.indices;
    if (nativeIndices) {
      // 检查一下内点的数量
      const indicesSize = nativeIndices.size();
      console.debug(`SACSegmentationFromNormals found ${indicesSize} inliers`);

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

export default SACSegmentationFromNormals;
