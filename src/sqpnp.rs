// Based on existing Rust & C++ implementations:
// - https://github.com/powei-lin/sqpnp_simple/blob/main/src/lib.rs
// - https://github.com/terzakig/sqpnp/blob/master/sqpnp/sqpnp.cpp

use std::marker::PhantomData;

use glam::{Mat3, Vec2, Vec3};
use nalgebra::{Const, Matrix, Matrix3, SMatrix, SVector, Vector2, Vector3};

type NVec2 = Vector2<f32>;
type NVec3 = Vector3<f32>;
type NVec6 = SVector<f32, 6>;
pub type NVec9 = SVector<f32, 9>;
pub type NMat3 = Matrix3<f32>;
pub type NMat9 = SMatrix<f32, 9, 9>;
type NMat3x9 = SMatrix<f32, 3, 9>;
type NMat6 = SMatrix<f32, 6, 6>;
type NMat9x3 = SMatrix<f32, 9, 3>;
type NMat9x6 = SMatrix<f32, 9, 6>;

fn vec9_to_mat3(v: &NVec9) -> NMat3 {
    v.reshape_generic(Const::<3>, Const::<3>)
}

fn mat3_to_vec9(v: &NMat3) -> NVec9 {
    v.reshape_generic(Const::<9>, Const::<1>)
}

const SQRT3: f32 = 1.73205080757;

// Solve `A*x=b` for 3x3 SPD `a`.
fn axb_solve_ldlt_3x3(a: &NMat3, b: &NVec3) -> Option<NVec3> {
    const EPSILON: f32 = 1e-10;

    let mut l = NMat3::zeros();

    l[(0, 0)] = a[(0, 0)];
    if l[(0, 0)] < EPSILON {
        return None;
    }
    let v = 1. / a[(0, 0)];
    l[(1, 0)] = a[(1, 0)] * v;
    l[(2, 0)] = a[(2, 0)] * v;

    let v = l[(1, 0)] * l[(0, 0)];
    l[(1, 1)] = a[(1, 1)] - l[(1, 0)] * v;
    if l[(1, 1)] < EPSILON {
        return None;
    }
    l[(2, 1)] = (a[(2, 1)] - l[(2, 0)] * v) / l[(1, 1)];

    let x = l[(2, 0)] * l[(0, 0)];
    let y = l[(2, 1)] * l[(1, 1)];
    l[(2, 2)] = a[(2, 2)] - l[(2, 0)] * x - l[(2, 1)] * y;
    if l[(2, 2)] < EPSILON {
        return None;
    }

    let x = b.x;
    let y = b.y - l[(1, 0)] * x;
    let z = b.z - l[(2, 0)] * x - l[(2, 1)] * y;

    let z = z / l[(2, 2)];
    let y = y / l[(1, 1)] - l[(2, 1)] * z;
    let x = x / l[(0, 0)] - l[(1, 0)] * y - l[(2, 0)] * z;

    Some(NVec3::new(x, y, z))
}

fn orthogonality_error(x: &NMat3) -> f32 {
    let a: NVec3 = x.column(0).into();
    let b: NVec3 = x.column(1).into();
    let c: NVec3 = x.column(2).into();

    let la = a.magnitude_squared();
    let lb = b.magnitude_squared();
    let lc = c.magnitude_squared();

    let dab = a.dot(&b);
    let dac = a.dot(&c);
    let dbc = b.dot(&c);

    (la - 1.) * (la - 1.)
        + (lb - 1.) * (lb - 1.)
        + (lc - 1.) * (lc - 1.)
        + 2. * dab * dab * dac * dac * dbc * dbc
}

#[derive(Clone, Debug, Default)]
pub struct SqPnPSolution {
    rotation: NVec9,
    translation: NVec3,
    num_iterations: usize,
    sq_error: f32,
}

impl SqPnPSolution {
    pub fn rotation_matrix(&self) -> Mat3 {
        let mat: Mat3 = self.rotation
            .reshape_generic(Const::<3>, Const::<3>)
            .into();
        mat.transpose()
    }

    pub fn translation(&self) -> Vec3 {
        self.translation.into()
    }

    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    pub fn sq_error(&self) -> f32 {
        self.sq_error
    }

    fn fast_transform_z(&self, p: NVec3) -> f32 {
        let r = &self.rotation;
        let t = &self.translation;
        r[6] * p.x + r[7] * p.y * r[8] * p.z + t.z
    }

    fn has_positive_depth(&self, p: NVec3) -> bool {
        self.fast_transform_z(p) > 0.
    }

    fn has_positive_majority_depths(&self, points: &[Vec3], weights: Option<&[f32]>) -> bool {
        let (pos, neg) = if let Some(weights) = weights {
            points.iter()
                .zip(weights)
                .fold((0, 0), |(pos, neg), (&p, &w)| {
                    if w <= 0.0 {
                        (pos, neg)
                    } else {
                        let z = self.fast_transform_z(p.into());
                        if z > 0. {
                            (pos + 1, neg)
                        } else {
                            (pos, neg + 1)
                        }
                    }
                })
        } else {
            points.iter()
                .fold((0, 0), |(pos, neg), &p| {
                    let z = self.fast_transform_z(p.into());
                    if z > 0. {
                        (pos + 1, neg)
                    } else {
                        (pos, neg + 1)
                    }
                })
        };

        pos >= neg
    }
}

pub trait SqPnPParameters {
    const RANK_TOLERANCE: f32;
    const SQP_SQUARED_TOLERANCE: f32;
    const SQP_DET_THRESHOLD: f32;
    const SQP_MAX_ITERATIONS: usize;
    const ORTHOGONALITY_SQUARED_ERROR_THRESHOLD: f32;
    const EQUAL_VECTORS_SQUARED_DIFF: f32;
    const EQUAL_SQUARED_ERRORS_DIFF: f32;
    const POINT_VARIANCE_THRESHOLD: f32;

    fn nearest_rotation_matrix(m: &NMat3) -> NMat3;
    fn eigen_vectors(m: &NMat9) -> (NMat9, NVec9);
}

pub fn nearest_rotation_matrix_svd(m: &NMat3) -> NMat3 {
    let svd = m.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let det_uv = u.determinant() * v_t.determinant();
    let diagonal = NVec3::new(1., 1., det_uv);
    let r = u * Matrix::from_diagonal(&diagonal) * v_t;
    r
}

pub fn nearest_rotation_matrix_foam(m: &NMat3) -> NMat3 {
    let det = m.determinant();
    if det.abs() < 1e-4 {
        return nearest_rotation_matrix_svd(m);
    }

    let n = m.adjoint();
    let lm = m.magnitude_squared();
    let ln = n.magnitude_squared();

    let mut l = 1.5 + 0.5 * lm;
    if det < 0. {
        l = -l;
    }

    let mut l_prev = 0.;
    for _ in 0..15 {
        if (l - l_prev).abs() <= 1e-12 * l_prev.abs() {
            break;
        }

        let x = l * l - lm;
        let p = x * x - 8. * l * det - 4. * ln;
        let pp = 8. * (0.5 * x * l - det);
        l_prev = l;
        l -= p / pp;
    }

    let mmt = m * m.transpose();
    let o = mmt * m;
    let a = l * l + lm;
    let denom = l * (l * l - lm) - 2. * det;
    let denom = denom.recip();
    let r = ((a * m) + 2. * (l * n - o)) * denom;
    r
}

pub fn eigen_vectors_rrqr(m: &NMat9) -> (NMat9, NVec9) {
    let qr = m.qr();
    let q = qr.q();
    let r = qr.unpack_r().diagonal().abs();
    (q, r)
}

pub fn eigen_vectors_svd(m: &NMat9) -> (NMat9, NVec9) {
    let svd = m.svd(true, false);
    let u = svd.u.unwrap();
    let s = svd.singular_values;
    (u, s)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultSqPnPParameters;

impl SqPnPParameters for DefaultSqPnPParameters {
    const RANK_TOLERANCE: f32 = 1e-7;
    const SQP_SQUARED_TOLERANCE: f32 = 1e-10;
    const SQP_DET_THRESHOLD: f32 = 1.01;
    const SQP_MAX_ITERATIONS: usize = 15;
    const ORTHOGONALITY_SQUARED_ERROR_THRESHOLD: f32 = 1e-8;
    const EQUAL_VECTORS_SQUARED_DIFF: f32 = 1e-10;
    const EQUAL_SQUARED_ERRORS_DIFF: f32 = 1e-6;
    const POINT_VARIANCE_THRESHOLD: f32 = 1e-5;

    fn nearest_rotation_matrix(m: &NMat3) -> NMat3 {
        // nearest_rotation_matrix_noam(m)
        nearest_rotation_matrix_svd(m)
    }

    fn eigen_vectors(m: &NMat9) -> (NMat9, NVec9) {
        // eigen_vectors_rrqr(m)
        eigen_vectors_svd(m)
    }
}

#[derive(Clone, Debug)]
pub struct SqPnPSolver<P = DefaultSqPnPParameters> {
    _parameters: PhantomData<P>,
    solutions: Vec<SqPnPSolution>,
}

impl<P> Default for SqPnPSolver<P> {
    fn default() -> Self {
        SqPnPSolver {
            _parameters: PhantomData,
            solutions: Vec::new(),
        }
    }
}

impl<P: SqPnPParameters> SqPnPSolver<P> {
    pub fn solutions(&self) -> &[SqPnPSolution] {
        &self.solutions
    }

    pub fn new() -> SqPnPSolver<P> {
        SqPnPSolver::default()
    }

    fn nearest_rotation_matrix(r: &NVec9) -> NVec9 {
        mat3_to_vec9(&P::nearest_rotation_matrix(&vec9_to_mat3(&r)))
    }

    fn handle_solution(
        &mut self,
        mut solution: SqPnPSolution,
        mut min_sq_err: f32,
        points: &[Vec3],
        weights: Option<&[f32]>,
        mean: NVec3,
        omega: &NMat9,
    ) -> f32 {
        if !solution.has_positive_depth(mean)
            && !solution.has_positive_majority_depths(points, weights) {
            return min_sq_err;
        }

        let old_sq_err = min_sq_err;
        let new_sq_err = (omega * solution.rotation)
            .dot(&solution.rotation);
        solution.sq_error = new_sq_err;
        if min_sq_err > new_sq_err {
            min_sq_err = new_sq_err;
        }

        if (new_sq_err - old_sq_err).abs() > P::EQUAL_SQUARED_ERRORS_DIFF {
            if old_sq_err > new_sq_err {
                self.solutions.clear();
                self.solutions.push(solution);
            }
            return min_sq_err;
        }

        for existing in &mut self.solutions {
            let e = (existing.rotation - solution.rotation)
                .magnitude_squared();
            if e >= P::EQUAL_VECTORS_SQUARED_DIFF {
                continue;
            }

            if existing.sq_error > new_sq_err {
                *existing = solution;
                return min_sq_err;
            }
        }

        self.solutions.push(solution);
        min_sq_err
    }

    fn row_and_null_space(
        r: &NVec9,
    ) -> (NMat9x6, NMat9x3, NMat6) {
        const NORM_THRESHOLD: f32 = 0.1;

        let mut h = NMat9x6::zeros();
        let mut k = NMat6::zeros();

        let a: NVec3 = r.fixed_rows::<3>(0).into();
        let b: NVec3 = r.fixed_rows::<3>(3).into();
        let c: NVec3 = r.fixed_rows::<3>(6).into();

        let la = a.magnitude();
        let ila = if la > 1e-5 { la.recip() } else { 0. };
        let q1 = a * ila;
        let m1 = 2. * la;
        h[(0, 0)] = q1[0];
        h[(1, 0)] = q1[1];
        h[(2, 0)] = q1[2];
        k[(0, 0)] = m1;

        let lb = b.magnitude();
        let ilb = lb.recip();
        let q2 = b * ilb;
        let m2 = 2. * lb;
        h[(3, 1)] = q2[0];
        h[(4, 1)] = q2[1];
        h[(5, 1)] = q2[2];
        k[(1, 1)] = m2;

        let lc = c.magnitude();
        let ilc = lc.recip();
        let q3 = c * ilc;
        let m3 = 2. * lc;
        h[(6, 2)] = q3[0];
        h[(7, 2)] = q3[1];
        h[(8, 2)] = q3[2];
        k[(2, 2)] = m3;

        let db1 = b.dot(&q1);
        let da2 = a.dot(&q2);
        let q4 = b - db1 * q1;
        let q5 = a - da2 * q2;
        let il34 = (q4.magnitude_squared() + q5.magnitude_squared()).sqrt().recip();
        let q4 = q4 * il34;
        let q5 = q5 * il34;
        h[(0, 3)] = q4[0];
        h[(1, 3)] = q4[1];
        h[(2, 3)] = q4[2];
        h[(3, 3)] = q5[0];
        h[(4, 3)] = q5[1];
        h[(5, 3)] = q5[2];
        k[(3, 0)] = b.dot(&q1);
        k[(3, 1)] = a.dot(&q2);
        k[(3, 3)] = b.dot(&q4) + a.dot(&q5);

        let dc2 = c.dot(&q2);
        let db3 = b.dot(&q3);
        let dc4 = c.dot(&q4);
        let q6 = -dc4 * q4;
        let q7 = c - dc2 * q2 - dc4 * q5;
        let q8 = b - db3 * q3;
        let il8 = (q6.magnitude_squared() + q7.magnitude_squared() + q8.magnitude_squared())
            .sqrt()
            .recip();
        let q6 = q6 * il8;
        let q7 = q7 * il8;
        let q8 = q8 * il8;
        h[(0, 4)] = q6.x;
        h[(1, 4)] = q6.y;
        h[(2, 4)] = q6.z;
        h[(3, 4)] = q7.x;
        h[(4, 4)] = q7.y;
        h[(5, 4)] = q7.z;
        h[(6, 4)] = q8.x;
        h[(7, 4)] = q8.y;
        h[(8, 4)] = q8.z;
        k[(4, 1)] = c.dot(&q2);
        k[(4, 2)] = b.dot(&q3);
        k[(4, 3)] = c.dot(&q5);
        k[(4, 4)] = c.dot(&q7) + b.dot(&q8);

        let dc1 = c.dot(&q1);
        let da3 = a.dot(&q3);
        let dc4 = c.dot(&q4);
        let da8 = a.dot(&q8);
        let q9 = c - dc1 * q1 - dc4 * q4 - da8 * q6;
        let q10 = -da8 * q7 - dc4 * q5;
        let q11 = a - da3 * q3 - da8 * q8;
        let il11 = (q9.magnitude_squared() + q10.magnitude_squared() + q11.magnitude_squared())
            .sqrt()
            .recip();
        let q9 = q9 * il11;
        let q10 = q10 * il11;
        let q11 = q11 * il11;
        h[(0, 5)] = q9.x;
        h[(1, 5)] = q9.y;
        h[(2, 5)] = q9.z;
        h[(3, 5)] = q10.x;
        h[(4, 5)] = q10.y;
        h[(5, 5)] = q10.z;
        h[(6, 5)] = q11.x;
        h[(7, 5)] = q11.y;
        h[(8, 5)] = q11.z;
        k[(5, 0)] = c.dot(&q1);
        k[(5, 2)] = a.dot(&q3);
        k[(5, 3)] = c.dot(&q4);
        k[(5, 4)] = c.dot(&q6) + a.dot(&q8);
        k[(5, 5)] = c.dot(&q9) + a.dot(&q11);

        let pn = NMat9::identity() - h * h.transpose();
        let mut n = NMat9x3::zeros();
        let mut col_len = [
            pn.column(0).magnitude(),
            pn.column(1).magnitude(),
            pn.column(2).magnitude(),
            pn.column(3).magnitude(),
            pn.column(4).magnitude(),
            pn.column(5).magnitude(),
            pn.column(6).magnitude(),
            pn.column(7).magnitude(),
            pn.column(8).magnitude(),
        ];

        let (max_1_index, max_1_value) = col_len.iter()
            .enumerate()
            .fold(None, |acc, (index, &len)| {
                if len < NORM_THRESHOLD {
                    acc
                } else if let Some((prev_index, prev_len)) = acc {
                    if len > prev_len {
                        Some((index, len))
                    } else {
                        Some((prev_index, prev_len))
                    }
                } else {
                    Some((index, len))
                }
            })
            .unwrap();

        let v1: NVec9 = pn.column(max_1_index).into();
        n.set_column(0, &(v1 / max_1_value));
        col_len[max_1_index] = -1.;

        let (min_2_index, _) = col_len.iter()
            .enumerate()
            .fold(None, |acc, (index, &len)| {
                if len < NORM_THRESHOLD {
                    acc
                } else {
                    let value = (pn.column(index).dot(&v1) / len).abs();
                    if let Some((prev_index, prev_value)) = acc {
                        if value < prev_value {
                            Some((index, value))
                        } else {
                            Some((prev_index, prev_value))
                        }
                    } else {
                        Some((index, value))
                    }
                }
            })
            .unwrap();

        let v2: NVec9 = pn.column(min_2_index).into();
        let x = v2 - v2.dot(&n.column(0)) * n.column(0);
        let x = x / x.magnitude();
        n.set_column(1, &x);
        col_len[min_2_index] = -1.;

        let (min_3_index, _) = col_len.iter()
            .enumerate()
            .fold(None, |acc, (index, &len)| {
                if len < NORM_THRESHOLD {
                    acc
                } else {
                    let i = len.recip();
                    let a = (pn.column(index).dot(&v1) * i).abs();
                    let b = (pn.column(index).dot(&v2) * i).abs();
                    let value = a + b;
                    if let Some((prev_index, prev_value)) = acc {
                        if value < prev_value {
                            Some((index, value))
                        } else {
                            Some((prev_index, prev_value))
                        }
                    } else {
                        Some((index, value))
                    }
                }
            })
            .unwrap();

        let v3: NVec9 = pn.column(min_3_index).into();
        let x = v3
            - (v3.dot(&n.column(1)) * n.column(1))
            - (v3.dot(&n.column(0)) * n.column(0));
        let x = x / x.magnitude();
        n.set_column(2, &x);

        (h, n, k)
    }

    fn solve_sqp_system(r: &NVec9, omega: &NMat9) -> NVec9 {
        let a: NVec3 = r.fixed_rows::<3>(0).into();
        let b: NVec3 = r.fixed_rows::<3>(3).into();
        let c: NVec3 = r.fixed_rows::<3>(6).into();

        let la2 = a.magnitude_squared();
        let lb2 = b.magnitude_squared();
        let lc2 = c.magnitude_squared();

        let dab = a.dot(&b);
        let dac = a.dot(&c);
        let dbc = b.dot(&c);

        let (h, n, jh) = Self::row_and_null_space(r);

        let g = NVec6::from([1. - la2, 1. - lb2, 1. - lc2, -dab, -dbc, -dac]);
        let x0 = g[0] / jh[(0, 0)];
        let x1 = g[1] / jh[(1, 1)];
        let x2 = g[2] / jh[(2, 2)];
        let x3 = (g[3] - x0 * jh[(3, 0)] - x1 * jh[(3, 1)]) / jh[(3, 3)];
        let x4 = (g[4] - x1 * jh[(4, 1)] - x2 * jh[(4, 2)] - x3 * jh[(4, 3)]) / jh[(4, 4)];
        let x5 = (g[5] - x0 * jh[(5, 0)] - x2 * jh[(5, 2)] - x3 * jh[(5, 3)] - x4 * jh[(5, 4)]) / jh[(5, 5)];
        let x = NVec6::from([x0, x1, x2, x3, x4, x5]);

        let delta = h * x;
        let nt_omega = n.transpose() * omega;
        let w = nt_omega * n;
        let rhs = -(nt_omega * (delta + r));

        let y = if let Some(v) = axb_solve_ldlt_3x3(&w, &rhs) {
            v
        } else {
            let w_inv = w.try_inverse().unwrap();
            w_inv * rhs
        };

        delta + n * y
    }

    fn run_sqp(r0: &NVec9, omega: &NMat9) -> SqPnPSolution {
        let mut r = *r0;
        let mut delta_len_sq = f32::MAX;
        let mut step = 0;

        while delta_len_sq > P::SQP_SQUARED_TOLERANCE && step < P::SQP_MAX_ITERATIONS {
            step += 1;
            let delta = Self::solve_sqp_system(&r, omega);
            r += delta;
            delta_len_sq = delta.magnitude_squared();
        }

        let r_mat = vec9_to_mat3(&r);
        let mut det_r = r_mat.determinant();
        if det_r < 0. {
            r = -r;
            det_r = -det_r;
        }

        let clean_r = if det_r > P::SQP_DET_THRESHOLD {
            Self::nearest_rotation_matrix(&r)
        } else {
            r
        };

        SqPnPSolution {
            num_iterations: step,
            rotation: clean_r,
            ..Default::default()
        }
    }

    fn calculate_omega(
        points_3d: &[Vec3],
        points_2d: &[Vec2],
        weights: Option<&[f32]>,
    ) -> Option<(NMat9, NMat3x9, NVec3)> {
        let mut omega = NMat9::zeros();
        let mut qa = NMat3x9::zeros();

        let mut sum_w = 0.;
        let mut sum_w_proj = NVec2::zeros();
        let mut sum_w_proj_len_sq = 0.;
        let mut sum_w_p = NVec3::zeros();
        for (index, (&p3d, &p2d)) in points_3d.iter().zip(points_2d).enumerate() {
            let p3d = NVec3::from(p3d);
            let p2d = NVec2::from(p2d);
            let w = weights.map_or(1., |pts| pts[index]);
            let wp3d = w * p3d;
            let wp2d = w * p2d;
            let len_sq = p2d.magnitude_squared();
            let w_len_sq = w * len_sq;

            sum_w += w;
            sum_w_proj += wp2d;
            sum_w_proj_len_sq += w_len_sq;
            sum_w_p += wp3d;

            let px = wp3d.x * p3d;
            let py = wp3d.y * p3d;
            let pz = wp3d.z * p3d;

            omega[(0, 0)] += px.x;
            omega[(0, 1)] += px.y;
            omega[(0, 2)] += px.z;
            omega[(1, 1)] += py.y;
            omega[(1, 2)] += py.z;
            omega[(2, 2)] += pz.z;

            omega[(0, 6)] -= px.x * p2d.x;
            omega[(0, 7)] -= px.y * p2d.x;
            omega[(0, 8)] -= px.z * p2d.x;
            omega[(1, 7)] -= py.y * p2d.x;
            omega[(1, 8)] -= py.z * p2d.x;
            omega[(2, 8)] -= pz.z * p2d.x;

            omega[(3, 6)] -= px.x * p2d.y;
            omega[(3, 7)] -= px.y * p2d.y;
            omega[(3, 8)] -= px.z * p2d.y;
            omega[(4, 7)] -= py.y * p2d.y;
            omega[(4, 8)] -= py.z * p2d.y;
            omega[(5, 8)] -= pz.z * p2d.y;

            omega[(6, 6)] += len_sq * px.x;
            omega[(6, 7)] += len_sq * px.y;
            omega[(6, 8)] += len_sq * px.z;
            omega[(7, 7)] += len_sq * py.y;
            omega[(7, 8)] += len_sq * py.z;
            omega[(8, 8)] += len_sq * pz.z;

            qa[(0, 0)] += wp3d.x;
            qa[(0, 1)] += wp3d.y;
            qa[(0, 2)] += wp3d.z;
            qa[(0, 6)] -= wp3d.x * p2d.x;
            qa[(0, 7)] -= wp3d.y * p2d.x;
            qa[(0, 8)] -= wp3d.z * p2d.x;
            qa[(1, 6)] -= wp3d.x * p2d.y;
            qa[(1, 7)] -= wp3d.y * p2d.y;
            qa[(1, 8)] -= wp3d.z * p2d.y;
            qa[(2, 6)] += len_sq * wp3d.x;
            qa[(2, 7)] += len_sq * wp3d.y;
            qa[(2, 8)] += len_sq * wp3d.z;
        }

        // Complete qa
        qa[(1, 3)] = qa[(0, 0)];
        qa[(1, 4)] = qa[(0, 1)];
        qa[(1, 5)] = qa[(0, 2)];
        qa[(2, 0)] = qa[(0, 6)];
        qa[(2, 1)] = qa[(0, 7)];
        qa[(2, 2)] = qa[(0, 8)];
        qa[(2, 3)] = qa[(1, 6)];
        qa[(2, 4)] = qa[(1, 7)];
        qa[(2, 5)] = qa[(1, 8)];

        // Complete omega
        omega[(1, 6)] = omega[(0, 7)];
        omega[(2, 6)] = omega[(0, 8)];
        omega[(2, 7)] = omega[(1, 8)];
        omega[(4, 6)] = omega[(3, 7)];
        omega[(5, 6)] = omega[(3, 8)];
        omega[(5, 7)] = omega[(4, 8)];
        omega[(7, 6)] = omega[(6, 7)];
        omega[(8, 6)] = omega[(6, 8)];
        omega[(8, 7)] = omega[(7, 8)];

        omega[(3, 3)] = omega[(0, 0)];
        omega[(3, 4)] = omega[(0, 1)];
        omega[(3, 5)] = omega[(0, 2)];
        omega[(4, 4)] = omega[(1, 1)];
        omega[(4, 5)] = omega[(1, 2)];
        omega[(5, 5)] = omega[(2, 2)];

        omega[(1, 0)] = omega[(0, 1)];
        omega[(2, 0)] = omega[(0, 2)];
        omega[(2, 1)] = omega[(1, 2)];
        omega[(3, 0)] = omega[(0, 3)];
        omega[(3, 1)] = omega[(1, 3)];
        omega[(3, 2)] = omega[(2, 3)];
        omega[(4, 0)] = omega[(0, 4)];
        omega[(4, 1)] = omega[(1, 4)];
        omega[(4, 2)] = omega[(2, 4)];
        omega[(4, 3)] = omega[(3, 4)];
        omega[(5, 0)] = omega[(0, 5)];
        omega[(5, 1)] = omega[(1, 5)];
        omega[(5, 2)] = omega[(2, 5)];
        omega[(5, 3)] = omega[(3, 5)];
        omega[(5, 4)] = omega[(4, 5)];
        omega[(6, 0)] = omega[(0, 6)];
        omega[(6, 1)] = omega[(1, 6)];
        omega[(6, 2)] = omega[(2, 6)];
        omega[(6, 3)] = omega[(3, 6)];
        omega[(6, 4)] = omega[(4, 6)];
        omega[(6, 5)] = omega[(5, 6)];
        omega[(7, 0)] = omega[(0, 7)];
        omega[(7, 1)] = omega[(1, 7)];
        omega[(7, 2)] = omega[(2, 7)];
        omega[(7, 3)] = omega[(3, 7)];
        omega[(7, 4)] = omega[(4, 7)];
        omega[(7, 5)] = omega[(5, 7)];
        omega[(8, 0)] = omega[(0, 8)];
        omega[(8, 1)] = omega[(1, 8)];
        omega[(8, 2)] = omega[(2, 8)];
        omega[(8, 3)] = omega[(3, 8)];
        omega[(8, 4)] = omega[(4, 8)];
        omega[(8, 5)] = omega[(5, 8)];

        let q = NMat3::new(
            sum_w, 0., -sum_w_proj.x,
            0., sum_w, -sum_w_proj.y,
            -sum_w_proj.x, -sum_w_proj.y, sum_w_proj_len_sq,
        );
        let Some(q_inv) = q.try_inverse() else {
            return None;
        };

        let p = -q_inv * qa;
        omega += qa.transpose() * p;

        let mean = sum_w_p / sum_w;
        Some((omega, p, mean))
    }

    pub fn solve(
        &mut self,
        points_3d: &[Vec3],
        points_2d: &[Vec2],
        weights: Option<&[f32]>,
    ) -> bool {
        self.solutions.clear();
        if points_3d.len() != points_2d.len() || points_3d.len() < 4 {
            return false;
        }

        let Some((omega, p, point_mean)) =
            Self::calculate_omega(points_3d, points_2d, weights) else {
            return false;
        };
        let (eigen_vectors, eigen_values) = P::eigen_vectors(&omega);

        let mut num_null_vectors = 0;
        while num_null_vectors < 8 && eigen_values[7 - num_null_vectors] < P::RANK_TOLERANCE {
            num_null_vectors += 1;
        }

        if num_null_vectors > 6 {
            return false;
        }

        let mut min_sq_error = f32::MAX;
        let num_eigen_points = num_null_vectors.max(1);

        for i in 0..(9 - num_eigen_points) {
            let e = SQRT3 * eigen_vectors.column(i);
            let e_mat = vec9_to_mat3(&e);
            let o_err = orthogonality_error(&e_mat);

            if o_err < P::ORTHOGONALITY_SQUARED_ERROR_THRESHOLD {
                let det_e = e_mat.determinant();
                let r = e * det_e;
                let t = p * r;
                let solution = SqPnPSolution {
                    rotation: r,
                    translation: t,
                    ..Default::default()
                };
                min_sq_error = self.handle_solution(solution, min_sq_error, points_3d, weights, point_mean, &omega);
                continue;
            }

            let r = Self::nearest_rotation_matrix(&e);
            let mut solution = Self::run_sqp(&r, &omega);
            solution.translation = p * solution.rotation;
            min_sq_error = self.handle_solution(solution, min_sq_error, points_3d, weights, point_mean, &omega);

            let r = Self::nearest_rotation_matrix(&-e);
            let mut solution = Self::run_sqp(&r, &omega);
            solution.translation = p * solution.rotation;
            min_sq_error = self.handle_solution(solution, min_sq_error, points_3d, weights, point_mean, &omega);
        }

        let mut c = 1;
        loop {
            let index = 9 - num_eigen_points - c;
            if index == 0 {
                break;
            }

            if min_sq_error <= 3. * eigen_values[index] {
                break;
            }

            let e = eigen_vectors.column(index).into();
            let r = Self::nearest_rotation_matrix(&e);
            let mut solution = Self::run_sqp(&r, &omega);
            solution.translation = p * solution.rotation;
            min_sq_error = self.handle_solution(solution, min_sq_error, points_3d, weights, point_mean, &omega);

            let r = Self::nearest_rotation_matrix(&-e);
            let mut solution = Self::run_sqp(&r, &omega);
            solution.translation = p * solution.rotation;
            min_sq_error = self.handle_solution(solution, min_sq_error, points_3d, weights, point_mean, &omega);

            c += 1;
        }

        !self.solutions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{vec3, EulerRot, Quat};

    #[test]
    fn test_example_solution() {
        let p3d = [
            vec3(-1., -1., -1.),
            vec3(1., -1., -1.),
            vec3(-1., 1., -1.),
            vec3(1., 1., -1.),
            vec3(-1., -1., 1.),
            vec3(1., -1., 1.),
            vec3(-1., 1., 1.),
            vec3(1., 1., 1.),
        ];

        let (rx, ry, rz) = (30.0f32, 40.0f32, 50.0f32);
        // let (rx, ry, rz) = (0f32, 0f32, 0f32);
        let (rx, ry, rz) = (rx.to_radians(), ry.to_radians(), rz.to_radians());
        let r = Quat::from_euler(EulerRot::XYZ, rx, ry, rz);
        let r = Mat3::from_quat(r);
        let hx = r.x_axis.length();
        let hy = r.y_axis.length();
        let hz = r.z_axis.length();
        let t = vec3(10., 20., 10.);
        // let t = vec3(0., 0., 0.);

        let p3d_t = p3d.map(|p| (r * p) + t);
        let p2d = p3d_t.map(|p| p.truncate() / p.z);

        let mut solver = SqPnPSolver::<DefaultSqPnPParameters>::new();
        assert!(solver.solve(&p3d, &p2d, None));

        let solution = &solver.solutions()[0];
        let s_r = solution.rotation_matrix();
        let (rx, ry, rz) = s_r.to_euler(EulerRot::XYZ);
        let (rx, ry, rz) = (rx.to_degrees(), ry.to_degrees(), rz.to_degrees());
        let s_t = solution.translation();

        let ix = s_r.x_axis.length();
        let iy = s_r.y_axis.length();
        let iz = s_r.z_axis.length();
        let dab = s_r.x_axis.dot(s_r.y_axis);
        let dbc = s_r.y_axis.dot(s_r.z_axis);
        let dac = s_r.z_axis.dot(s_r.x_axis);
        let d = dab.max(dbc).max(dac);

        let diff = s_r * r.inverse();
        println!("diff = {diff}");
        println!("r = {r}");
        println!("s_r = {s_r}");

        println!("h = {hx},{hy},{hz}");
        println!("i = {ix},{iy},{iz} d = {d}");
        println!("r={rx},{ry},{rz}");
        println!("s_t={s_t}");

        for (index, (&m, &a)) in p3d.iter().zip(&p3d_t).enumerate() {
            let b = s_t + s_r * m;
            let delta = b - a;
            let d2 = delta.length_squared();

            if d2 > 0.5 {
                assert!(false, "point {index} too far, expected {a} got {b} (t={s_t} r={rx},{ry},{rz} ({r}))");
            }
        }
    }
}
