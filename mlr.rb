require "numo/narray"

class MLR

  class DataShapeError < ArgumentError
    def initialize(msg="Data shape must be consistent (dimensions vs coefficients & x vs y length)", exception_type="custom")
      @exception_type = exception_type
      super(msg)
    end
  end

  # For testing
  def generate_dataset(n)
    x = []
    y = []
    random_x1 = rand()
    random_x2 = rand()
    (0..n-1).each do | i |
      x1 = i
      x2 = i/2 + rand()*n
      x.append([1, x1, x2])
      y.append(random_x1 * x1 + random_x2 * x2 + 1)
    end
    return Numo::DFloat.cast(x), Numo::DFloat.cast(y)
  end

  def mse(coef, x, y)
    return ((x.dot(coef) - y)**2).mean/2
  end


  def gradients(coef, x, y)
    return (x.transpose()*(x.dot(coef) - y)).mean(axis: 1)
  end


  def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8)

    unless x.shape[0] == y.shape[0] && x.shape[1] == coef.shape[0]
      raise DataShapeError
    end

    prev_error = 0
    m_coef = Numo::DFloat.zeros(coef.shape)
    v_coef = Numo::DFloat.zeros(coef.shape)
    moment_m_coef = Numo::DFloat.zeros(coef.shape)
    moment_v_coef = Numo::DFloat.zeros(coef.shape)
    t = 0

    loop do
      error = mse(coef, x, y)

      return coef if (error - prev_error).abs <= epsilon

      prev_error = error
      grad = gradients(coef, x, y)
      t += 1
      m_coef = b1 * m_coef + (1-b1)*grad
      v_coef = b2 * v_coef + (1-b2)*grad**2
      moment_m_coef = m_coef / (1-b1**t)
      moment_v_coef = v_coef / (1-b2**t)

      delta = ((lr / moment_v_coef**0.5 + 1e-8) *
               (b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))

      coef = coef - delta
    end
  end
end

mlr = MLR.new()

x, y = mlr.generate_dataset(200)

coef = Numo::DFloat.cast([0, 0, 0])
c = mlr.multilinear_regression(coef, x, y, 1e-1)



